import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_vlm import QwenVL, VLMConfig

warnings.filterwarnings('ignore', category=UserWarning)

QWEN_PATH = '../models/Qwen/Qwen2.5-0.5B-Instruct'
SIGLIP_PATH = '../models/google/siglip2-base-patch16-224'


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.bfloat16):
    VLMConfig.register_for_auto_class()
    QwenVL.register_for_auto_class("AutoModelForCausalLM")
    lm_model = QwenVL(lm_config, vision_model_path=SIGLIP_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'Model params: {model_params / 1e6:.2f}M = {model_params / 1e9:.3f}B')
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(transformers_path)

    config_path = os.path.join(transformers_path, "config.json")
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    config['tie_word_embeddings'] = True

    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        tcfg = json.load(open(tokenizer_config_path, 'r', encoding='utf-8'))
        tcfg.update({"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}})
        json.dump(tcfg, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Model saved as Transformers format: {transformers_path}")


if __name__ == '__main__':
    lm_config = VLMConfig(hidden_size=896, num_hidden_layers=24)
    torch_path = f"../out/sft_vlm_{lm_config.hidden_size}.pth"
    transformers_path = '../qwen-vl'
    convert_torch2transformers(torch_path, transformers_path)
