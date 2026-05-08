import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_vlm import QwenVL, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode,
    setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn
)

warnings.filterwarnings('ignore')

QWEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Qwen/Qwen2.5-0.5B-Instruct'))


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "learning_rate": current_lr, "eta": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # Merge LoRA into base model so checkpoint can be loaded without PeftModel
            if hasattr(raw_model.model, 'merge_and_unload'):
                raw_model.model = raw_model.model.merge_and_unload()
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V SFT (Qwen2.5-0.5B + SigLIP2)")
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="Weight prefix")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (for LoRA)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision dtype")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading threads")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument('--hidden_size', default=896, type=int, help="Hidden size (Qwen2.5-0.5B=896)")
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="Number of layers (Qwen2.5-0.5B=24)")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Max sequence length")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE (not used with Qwen)")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet", help="Training data path")
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str,
                        help="Pretrained weight prefix to load (none = skip)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto resume (0=no, 1=yes)")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2],
                        help="Freeze strategy (0=full ft, 1=LoRA, 2=only projector)")
    parser.add_argument('--max_samples', default=None, type=int, help="Limit training samples (None = all)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Apply LoRA to Qwen attention layers in SFT")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="Use torch.compile")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-SFT-Qwen", help="Wandb project name")
    args = parser.parse_args()

    # ========== 1. Init environment ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. Config and dirs ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
    )
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight,
                              save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. Mixed precision ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. Wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-SFT-Qwen-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. Model, tokenizer, data ==========
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config, from_weight=args.from_weight, device=args.device,
        freeze_llm=args.freeze_llm, use_lora=args.use_lora,
    )
    train_ds = VLMDataset(
        args.data_path, tokenizer, preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        max_length=vlm_config.max_position_embeddings,
        max_samples=args.max_samples,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )

    # ========== 6. Resume ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. Compile & DDP ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. Training loop ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                           num_workers=args.num_workers, pin_memory=True, collate_fn=vlm_collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping first {start_step} steps')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
