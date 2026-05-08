"""对 RL dataset 中的每张图，使用数据集中的提示词，采样多个结果。

用法:
    # 从头开始，每张图采样 4 个结果
    python build_rl_dataset/sample.py

    # 多图并行，batch_size=8
    python build_rl_dataset/sample.py --batch_size 8

    # 从第 100 条开始（跳过前 100 条），用于中断后恢复
    python build_rl_dataset/sample.py --start 100

    # 只处理 50 条
    python build_rl_dataset/sample.py --start 100 --max_samples 50

    # 指定模型 checkpoint
    python build_rl_dataset/sample.py --weight sft_250k_10k_896

    # 自定义采样参数
    python build_rl_dataset/sample.py --temperature 0.8 --top_p 0.9 --max_new_tokens 256
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
import time
import warnings
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from model.model_vlm import QwenVL, VLMConfig

warnings.filterwarnings('ignore')


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

QWEN_PATH = os.path.abspath('./models/Qwen/Qwen2.5-0.5B-Instruct')
SIGLIP_PATH = os.path.abspath('./models/google/siglip2-base-patch16-224')


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH, trust_remote_code=True, local_files_only=True
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
    tokenizer.pad_token = tokenizer.eos_token

    ckp = os.path.join(args.save_dir, f'{args.weight}.pth')
    if not os.path.exists(ckp):
        raise FileNotFoundError(f'Checkpoint not found: {ckp}')

    model = QwenVL(
        VLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            vocab_size=len(tokenizer),
        ),
        vision_model_path=SIGLIP_PATH,
    )
    img_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    model.config.image_ids = [img_id]

    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(
        {k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False
    )

    model = model.half().eval().to(args.device)
    return model, tokenizer, model.processor


def load_dataset(jsonl_path, start, max_samples):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < start:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def build_prompt(conversations, model, tokenizer):
    role_map = {'human': 'user', 'gpt': 'assistant'}
    messages = []
    for msg in conversations:
        role = role_map.get(msg['from'], msg['from'])
        content = msg['value']
        content = content.replace(
            '<image>',
            model.config.image_special_token * model.config.image_token_len,
        )
        messages.append({'role': role, 'content': content})

    for msg in reversed(messages):
        if msg['role'] == 'user':
            msg['content'] = msg['content'] + '\n请尽量简短回答。'
            break

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def sample_batch(model, tokenizer, text_inputs, pixel_values, args):
    """Run batched generation: B images → B × num_return_sequences outputs.

    text_inputs: list of str, length B
    pixel_values: {'pixel_values': tensor (B, C, H, W)}
    Returns: list of list of str, shape (B, num_return_sequences)
    """
    # Tokenize with padding for batched inputs
    inputs = tokenizer(
        text_inputs, return_tensors='pt', padding=True, truncation=True
    ).to(args.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            num_return_sequences=args.num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            pixel_values=pixel_values,
        )

    # Unpack: outputs are interleaved as [img0_s0, img0_s1, ..., img0_sN-1, img1_s0, ...]
    B = len(text_inputs)
    N = args.num_samples
    all_responses = []
    for i in range(B):
        responses = []
        input_ids_for_this = inputs['input_ids'][i]
        for j in range(N):
            gen_ids = generated_ids[i * N + j]
            # Trim to only generated tokens (skip the input part)
            # All inputs were padded to same length, but each image has its own
            # actual length — we use attention_mask to find the real boundary
            gen_tokens = gen_ids[len(input_ids_for_this):]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            response = response.replace('\n', ' ').replace('\r', ' ')
            responses.append(response)
        all_responses.append(responses)

    return all_responses


def main():
    parser = argparse.ArgumentParser(description='Sample multiple responses per image for RL dataset')
    # Data
    parser.add_argument('--dataset_dir', default='rl_dataset/RLHF-V/dataset_2', type=str)
    parser.add_argument('--output', default=None, type=str,
                        help='Output JSONL file (default: <dataset_dir>/samples.jsonl)')
    # Model
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='sft_250k_10k_896', type=str)
    parser.add_argument('--hidden_size', default=896, type=int)
    parser.add_argument('--num_hidden_layers', default=24, type=int)
    # Sampling
    parser.add_argument('--num_samples', default=4, type=int, help='Samples per image')
    parser.add_argument('--batch_size', default=10, type=int, help='Images per batch')
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    # Progress control
    parser.add_argument('--start', default=0, type=int, help='Skip first N items (for resume)')
    parser.add_argument('--max_samples', default=None, type=int, help='Max items to process')
    # Device
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps', type=str)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.dataset_dir, 'samples.jsonl')

    print(f'Device: {args.device}', flush=True)
    print(f'Model: {args.save_dir}/{args.weight}.pth', flush=True)
    print(f'Dataset: {args.dataset_dir}', flush=True)
    print(f'Output: {args.output}', flush=True)
    print(f'Start index: {args.start}, Max samples: {args.max_samples or "all"}', flush=True)
    print(f'Batch size: {args.batch_size}, Samples per image: {args.num_samples}', flush=True)
    print(f'Effective batch: {args.batch_size * args.num_samples}', flush=True)
    print(flush=True)

    # Load model
    print('Loading model...', flush=True)
    model, tokenizer, preprocess = load_model(args)
    print('Model loaded.\n', flush=True)

    # Load dataset
    jsonl_path = os.path.join(args.dataset_dir, 'data.jsonl')
    records = load_dataset(jsonl_path, args.start, args.max_samples)
    total = len(records)
    print(f'Records to process: {total}\n', flush=True)

    # Open output file in append mode for incremental saving
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_file = open(args.output, 'a', encoding='utf-8')

    total_start = time.time()
    batch_count = 0

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_records = records[batch_start:batch_end]
        B = len(batch_records)
        batch_count += 1

        # Load images for this batch
        images = []
        text_inputs = []
        for record in batch_records:
            img_path = os.path.join(args.dataset_dir, record['image'])
            images.append(Image.open(img_path).convert('RGB'))
            text_inputs.append(build_prompt(record['conversations'], model, tokenizer))

        # Batch-process images through preprocessor
        pixel_values = {
            k: v.to(args.device)
            for k, v in preprocess(images=images, return_tensors='pt').items()
        }

        st = time.time()
        all_responses = sample_batch(model, tokenizer, text_inputs, pixel_values, args)
        elapsed = time.time() - st

        # Save results for each image in the batch
        for j, (record, responses) in enumerate(zip(batch_records, all_responses)):
            global_idx = args.start + batch_start + j
            result = {
                'id': global_idx,
                'image': record['image'],
                'conversations': record['conversations'],
                'samples': responses,
                'elapsed': round(elapsed, 2),
                'batch_idx': j,
                'batch_size': B,
            }
            out_file.write(json.dumps(result, ensure_ascii=False) + '\n')

        out_file.flush()

        # Progress
        done = batch_end
        avg_time = (time.time() - total_start) / batch_count
        eta = avg_time * (total // args.batch_size + (1 if total % args.batch_size else 0) - batch_count)
        print(f'[{done}/{total}]  batch={batch_start}-{batch_end - 1}  '
              f'time={elapsed:.1f}s  speed={elapsed / B:.1f}s/img  ETA={eta:.0f}s  '
              f'resp_lens={[[len(r) for r in resp] for resp in all_responses]}', flush=True)

    out_file.close()
    total_elapsed = time.time() - total_start
    print(f'\nDone. {total} items in {total_elapsed:.0f}s '
          f'(avg {total_elapsed / total:.1f}s/item)', flush=True)


if __name__ == '__main__':
    main()
