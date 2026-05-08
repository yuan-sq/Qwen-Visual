import time
import argparse
import os
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import QwenVL, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')

QWEN_PATH = os.path.abspath('./models/Qwen/Qwen2.5-0.5B-Instruct')
SIGLIP_PATH = os.path.abspath('./models/google/siglip2-base-patch16-224')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
    tokenizer.pad_token = tokenizer.eos_token

    if 'model' in args.load_from:
        ckp = f'./{args.save_dir}/{args.weight}.pth'
        model = QwenVL(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, vocab_size=len(tokenizer)),
            vision_model_path=SIGLIP_PATH,
        )
        # Set image token ID in config
        img_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        model.config.image_ids = [img_id]

        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = QwenVL.get_vision_model(SIGLIP_PATH)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer, model.processor


def main():
    parser = argparse.ArgumentParser(description="Qwen-VLM Chat")
    parser.add_argument('--load_from', default='model', type=str, help="Model load path (model=torch weights, other=transformers)")
    parser.add_argument('--save_dir', default='out', type=str, help="Weight directory")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="Weight prefix")
    parser.add_argument('--hidden_size', default=896, type=int, help="Hidden size")
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="Number of layers")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="Max new tokens")
    parser.add_argument('--temperature', default=0.7, type=float, help="Temperature")
    parser.add_argument('--top_p', default=0.85, type=float, help="Top-p sampling")
    parser.add_argument('--image_dir', default='./eval_images/', type=str, help="Image directory")
    parser.add_argument('--show_speed', default=1, type=int, help="Show decode speed")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device")
    parser.add_argument('--open_thinking', default=0, type=int, help="Enable thinking mode")
    parser.add_argument('--output', default='', type=str, help="Save results to file (default: <save_dir>/eval_<weight>.txt)")
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if not args.output:
        args.output = f'./{args.save_dir}/eval_{args.weight}.txt'
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_file = open(args.output, 'w', encoding='utf-8')

    prompt = "<image>\n简要描述图片内容"
    for image_file in sorted(os.listdir(args.image_dir)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        setup_seed(random.randint(1, 31415926))
        image_path = os.path.join(args.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        pixel_values = {k: v.to(args.device) for k, v in QwenVL.image2tensor(image, preprocess).items()}

        messages = [{"role": "user", "content": prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
        inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

        print(f'[Image]: {image_file}')
        print(f"💬: {repr(prompt)}")
        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values,
        )
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        elapsed = time.time() - st
        # Collect generated text for file output
        gen_text = tokenizer.decode(generated_ids[0][-gen_tokens:], skip_special_tokens=True)
        out_file.write(f'[Image]: {image_file}\n')
        out_file.write(f'Prompt: {prompt}\n')
        out_file.write(f'Response: {gen_text}\n')
        out_file.write(f'Speed: {gen_tokens / elapsed:.2f} tokens/s\n\n')

        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / elapsed:.2f} tokens/s\n\n')
        else:
            print('\n\n')

    out_file.close()
    print(f'Results saved to {args.output}')


if __name__ == "__main__":
    main()
