"""
Training utility functions
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


QWEN_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/Qwen/Qwen2.5-0.5B-Instruct'))
SIGLIP_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/google/siglip2-base-patch16-224'))


def init_vlm_model(vlm_config, from_weight='none', tokenizer_path=QWEN_PATH,
                   vision_model_path=SIGLIP_PATH, save_dir='../out',
                   device='cuda', freeze_llm=2, use_lora=False,
                   freeze_projector=False):
    # Load Qwen2.5-0.5B-Instruct
    qwen = Qwen2ForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    # Load tokenizer and add image placeholder token
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
    tokenizer.pad_token = tokenizer.eos_token

    # Resize embeddings to accommodate the new token
    qwen.resize_token_embeddings(len(tokenizer))

    # Record image token ID in config
    img_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vlm_config.image_ids = [img_id]

    from model.model_vlm import QwenVL
    model = QwenVL(vlm_config, qwen_model=qwen, vision_model_path=vision_model_path)

    # Load pretrained VLM checkpoint if requested
    if from_weight != 'none':
        from model.model_vlm import VLMConfig
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        Logger(f'Loaded weights from {weight_path}')

    # Freeze vision encoder always (SigLIP2 stays frozen in both phases)
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # ===== Freeze strategies =====
    # freeze_llm == 2 : Pretrain — freeze Qwen + freeze SigLIP2, only train projector
    # freeze_llm == 1 : SFT — freeze SigLIP2, train projector + Qwen LoRA
    # freeze_llm == 0 : Full fine-tune (rare)

    if freeze_llm == 2:
        # Freeze every Qwen parameter
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # Projector is trainable by default (nn.Module default)

    elif freeze_llm == 1:
        # Freeze all Qwen params first
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # Projector stays trainable

        # Apply LoRA to Qwen attention layers
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.1,
                bias='none',
            )
            model.model = get_peft_model(model.model, lora_config)
            Logger('LoRA applied to Qwen attention layers')

    elif freeze_llm == 0:
        pass  # everything trainable

    if freeze_projector:
        for param in model.vision_proj.parameters():
            param.requires_grad = False
        Logger('Projector frozen')

    get_model_params(model, vlm_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable Params: {trainable:.3f}M')

    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None,
                   epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU count changed ({saved_ws}->{current_ws}), step adjusted to {ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
