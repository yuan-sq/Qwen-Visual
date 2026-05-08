"""对 RL dataset 中的问题+答案，用 LLM 一次性生成：
- 中文问题翻译
- 简洁中文 chosen
- 直译中文 rejected（golden answer 的直接翻译，可适当扩充）

用法:
    python build_rl_dataset/label_rl_dataset.py --dataset_dir rl_dataset/RLHF-V/dataset_1
    python build_rl_dataset/label_rl_dataset.py --dataset_dir rl_dataset/RLHF-V/dataset_2 --start 100 --max_samples 50
    python build_rl_dataset/label_rl_dataset.py --dataset_dir rl_dataset/RLHF-V/dataset_1 --workers 8
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import chat_json

PROMPT_TEMPLATE = """你正在为视觉问答 DPO 数据构造中文标注。给定一个关于图片的问题和 golden answer，请完成以下任务并输出 JSON。

任务：
1. 将问题翻译成中文（保留 <image> 标记在开头）。
2. 根据 golden answer 改写为简洁、直接、完整的中文 chosen。
3. 将 golden answer 直译为中文 rejected（可适当扩充句子使其通顺完整，但必须保留原意和所有事实细节）。

规则：
- chosen 必须简洁：删除不必要的解释、推测、背景描述和重复表达。
- yes/no 问题以"是"或"否"开头。数量问题直接回答。颜色/物体/位置问题用短语回答。
- 描述类问题 chosen 保留 1–2 句，rejected 保留 golden answer 的全部细节。
- 你的回复必须以 {{ 开头，以 }} 结尾，只包含有效的 JSON。严禁输出思考过程、解释说明或任何非 JSON 内容。

示例 1

问题：
<image>Is the body of water in the image an ocean or a lake?

Golden answer：
The body of water in the image appears to be a lake because the surface is calm and it seems to be surrounded by land.

输出：
{{"question_cn": "<image>图中的水体是海洋还是湖泊？", "chosen": "看起来是湖泊。", "rejected": "图中的水体看起来是一个湖泊，因为水面平静且似乎被陆地环绕。"}}

示例 2

问题：
<image>Please describe the image.

Golden answer：
There is an orange cat sitting on a windowsill, with several green plants beside it. Bright sunlight and some buildings can be seen outside the window, making the scene look warm and quiet.

输出：
{{"question_cn": "<image>请描述这张图片。", "chosen": "一只橘色的猫坐在窗台上，旁边有几盆绿色植物。", "rejected": "一只橘色的猫坐在窗台上，旁边有几盆绿色植物。窗外可以看到明亮的阳光和一些建筑物，场景温馨而安静。"}}

现在请处理：

问题：
{question}

Golden answer：
{golden_answer}

输出："""


def extract_question(record: dict) -> str:
    for msg in record.get('conversations', []):
        if msg.get('from') == 'human':
            return msg['value']
    return ''


def extract_golden_answer(record: dict) -> str:
    return record.get('chosen', {}).get('value', '')


def _find_json_braces(text: str) -> str | None:
    """Find the outermost JSON object using brace counting."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_output(text: str) -> dict | None:
    """Extract JSON from LLM output, handling various formats."""
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON block via brace counting
    json_str = _find_json_braces(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    # Last resort: find any {...} with required keys near each other
    m = re.search(r'\{"question_cn":\s*".+?"\s*,\s*"chosen":\s*".+?"\s*,\s*"rejected":\s*".+?"\}', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def label_one(record: dict) -> dict | None:
    """Label a single record. Returns updated record or None on failure."""
    question = extract_question(record)
    golden_answer = extract_golden_answer(record)

    if not question or not golden_answer:
        print(f'  [skip] missing data for {record.get("image", "?")}')
        return None

    prompt = PROMPT_TEMPLATE.format(question=question, golden_answer=golden_answer)

    for attempt in range(3):
        try:
            result = chat_json(prompt, temperature=0.3, max_tokens=1024, model='deepseek-chat')
            parsed = parse_json_output(result)
            if parsed and 'chosen' in parsed and 'rejected' in parsed:
                record['conversations'][0]['value'] = parsed.get('question_cn', question)
                record['chosen']['value'] = parsed['chosen']
                record['rejected']['value'] = parsed['rejected']
                return record
            else:
                print(f'  [parse] attempt {attempt+1}: {result[:120]!r}')
        except Exception as e:
            print(f'  [retry {attempt+1}/3] {e}')
        time.sleep(2 * (attempt + 1))

    print(f'  [fail] {record.get("image", "?")}')
    return None


def load_processed(output_path: str) -> set:
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    processed.add(d['image'])
                except json.JSONDecodeError:
                    continue
    return processed


def main():
    parser = argparse.ArgumentParser(description='Label RL dataset with Chinese Q/A via LLM')
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--max_samples', default=None, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--output', default=None, type=str,
                        help='Default: <dataset_dir>/labeled.jsonl')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.dataset_dir, 'labeled.jsonl')

    jsonl_path = os.path.join(args.dataset_dir, 'data.jsonl')
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < args.start:
                continue
            records.append(json.loads(line))
            if args.max_samples is not None and len(records) >= args.max_samples:
                break

    total = len(records)
    processed_set = load_processed(args.output)
    pending = [r for r in records if r['image'] not in processed_set]
    skipped = total - len(pending)

    print(f'Dataset: {args.dataset_dir}')
    print(f'Output: {args.output}')
    print(f'Total: {total}, Skipped: {skipped}, Pending: {len(pending)}')
    print(f'Workers: {args.workers}')
    print()

    if not pending:
        print('All done.')
        return

    out_file = open(args.output, 'a', encoding='utf-8')
    total_start = time.time()
    done = skipped

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(label_one, r): r for r in pending}

        for future in as_completed(future_map):
            done += 1
            record = future_map[future]
            try:
                result = future.result()
                if result is not None:
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_file.flush()
                    c = result['chosen']['value']
                    preview = c[:60] + '...' if len(c) > 60 else c
                    print(f'[{done}/{total}] {record["image"]} -> {preview}', flush=True)
                else:
                    print(f'[{done}/{total}] {record["image"]} FAILED', flush=True)
            except Exception as e:
                print(f'[{done}/{total}] {record["image"]} ERROR: {e}', flush=True)

    out_file.close()
    elapsed = time.time() - total_start
    print(f'\nDone. {done - skipped} records in {elapsed:.0f}s '
          f'(avg {elapsed / max(done - skipped, 1):.1f}s/record)', flush=True)


if __name__ == '__main__':
    main()
