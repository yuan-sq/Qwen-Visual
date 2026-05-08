import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """发送单轮对话，返回模型原始回复文本。"""
    try:
        response = client.chat.completions.create(
            model=model or MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"thinking": {"type": "enabled"}},
        )
        # 优先取 reasoning_content（thinking 内容），没有则取 content
        choice = response.choices[0]
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            return choice.message.reasoning_content
        return choice.message.content or ""
    except Exception as e:
        return f"[ERROR] {e}"


def parse_json_response(text: str) -> dict:
    """从模型回复中提取JSON。尝试完整解析，失败则正则匹配。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 {...} 块
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def chat_json(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """发送单轮对话并尝试解析 JSON 返回。失败则返回原始文本。"""
    text = chat(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    parsed = parse_json_response(text)
    if parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return text


def batch_chat(prompts: list[str], model: Optional[str] = None, max_tokens: int = 200, delay: float = 0.5) -> list[str]:
    """批量发送对话请求，每个prompt独立调用。"""
    results = []
    for i, prompt in enumerate(prompts):
        results.append(chat(prompt, model=model, max_tokens=max_tokens))
        if (i + 1) % 10 == 0:
            time.sleep(delay)  # 简单速率控制
    return results
