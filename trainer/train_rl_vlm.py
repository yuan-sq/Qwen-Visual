import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model.model_vlm import VLMConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_vlm_model,
    is_main_process,
    setup_seed,
    vlm_checkpoint,
)

warnings.filterwarnings('ignore')


class RLHFDataset(Dataset):
    """RLHF-V chosen/rejected dataset for offline DPO training."""

    def __init__(
        self,
        data_path,
        tokenizer,
        preprocess,
        image_special_token,
        image_token_len,
        max_length=1024,
        max_samples=None,
        image_root=None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.image_special_token = image_special_token
        self.image_token_len = image_token_len
        self.max_length = max_length
        self.image_root = image_root or os.path.dirname(data_path)
        self.records = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if 'chosen' in record and 'rejected' in record and 'image' in record:
                    self.records.append(record)
                if max_samples is not None and len(self.records) >= max_samples:
                    break

    def __len__(self):
        return len(self.records)

    def _prompt_text(self, conversations):
        role_map = {'human': 'user', 'gpt': 'assistant'}
        messages = []
        for msg in conversations:
            role = role_map.get(msg.get('from'), msg.get('from'))
            content = msg.get('value', '').replace(
                '<image>',
                self.image_special_token * self.image_token_len,
            )
            messages.append({'role': role, 'content': content})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _encode_pair(self, prompt_text, answer_text):
        eos = self.tokenizer.eos_token or ''
        full_text = prompt_text + answer_text + eos
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        ).input_ids
        input_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels) and len(labels) > 0:
            labels[-1] = input_ids[-1]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = self._prompt_text(record['conversations'])
        chosen_ids, chosen_labels = self._encode_pair(prompt, record['chosen']['value'])
        rejected_ids, rejected_labels = self._encode_pair(prompt, record['rejected']['value'])

        image_path = os.path.join(self.image_root, record['image'])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.preprocess(images=image, return_tensors='pt')
        pixel_values = {k: v.squeeze(0) for k, v in pixel_values.items()}

        return chosen_ids, chosen_labels, rejected_ids, rejected_labels, pixel_values


def dpo_collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    max_len = max(max(item[0].numel(), item[2].numel()) for item in batch)

    def pad_tensor(x, value):
        if x.numel() >= max_len:
            return x[:max_len]
        return F.pad(x, (0, max_len - x.numel()), value=value)

    chosen_ids = torch.stack([pad_tensor(item[0], pad_id) for item in batch])
    chosen_labels = torch.stack([pad_tensor(item[1], -100) for item in batch])
    rejected_ids = torch.stack([pad_tensor(item[2], pad_id) for item in batch])
    rejected_labels = torch.stack([pad_tensor(item[3], -100) for item in batch])

    pixel_data = [item[4] for item in batch]
    pixel_values = {
        key: torch.stack([data[key] for data in pixel_data])
        for key in pixel_data[0].keys()
    }
    return chosen_ids, chosen_labels, rejected_ids, rejected_labels, pixel_values


def move_pixel_values(pixel_values, device):
    if isinstance(pixel_values, dict):
        return {k: v.to(device, non_blocking=True) for k, v in pixel_values.items()}
    return pixel_values.to(device, non_blocking=True)


def sequence_log_probs(model, input_ids, labels, pixel_values, average_log_prob=False, length_penalty=0.0):
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]
    loss_mask = target.ne(-100)
    safe_target = target.masked_fill(~loss_mask, 0)
    token_log_probs = logits.log_softmax(dim=-1).gather(
        dim=-1, index=safe_target.unsqueeze(-1)
    ).squeeze(-1)
    seq_lens = loss_mask.sum(dim=-1)
    log_probs = (token_log_probs * loss_mask).sum(dim=-1)
    if average_log_prob:
        log_probs = log_probs / seq_lens.clamp(min=1)
    if length_penalty != 0:
        log_probs = log_probs - length_penalty * seq_lens
    return log_probs


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    policy_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen - ref_chosen).detach()
    rejected_rewards = beta * (policy_rejected - ref_rejected).detach()
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def save_train_state(epoch, step, wandb=None):
    vlm_checkpoint(
        vlm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir='../checkpoints',
        scaler=scaler,
    )


def save_eval_weights():
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    if hasattr(raw_model.model, 'merge_and_unload'):
        raw_model.model = raw_model.model.merge_and_unload()

    state_dict = raw_model.state_dict()
    clean_state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith('vision_encoder.')
    }
    clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
    ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.pth'
    torch.save(clean_state_dict, ckp)
    Logger(f'Saved eval weights to {ckp}')
    del state_dict, clean_state_dict


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        chosen_ids, chosen_labels, rejected_ids, rejected_labels, pixel_values = batch
        chosen_ids = chosen_ids.to(args.device, non_blocking=True)
        chosen_labels = chosen_labels.to(args.device, non_blocking=True)
        rejected_ids = rejected_ids.to(args.device, non_blocking=True)
        rejected_labels = rejected_labels.to(args.device, non_blocking=True)
        pixel_values = move_pixel_values(pixel_values, args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            policy_chosen = sequence_log_probs(
                model, chosen_ids, chosen_labels, pixel_values, args.average_log_prob, args.length_penalty
            )
            policy_rejected = sequence_log_probs(
                model, rejected_ids, rejected_labels, pixel_values, args.average_log_prob, args.length_penalty
            )
            with torch.no_grad():
                ref_chosen = sequence_log_probs(
                    ref_model, chosen_ids, chosen_labels, pixel_values, args.average_log_prob, args.length_penalty
                )
                ref_rejected = sequence_log_probs(
                    ref_model, rejected_ids, rejected_labels, pixel_values, args.average_log_prob, args.length_penalty
                )
            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta
            )
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
            reward_margin = (chosen_reward - rejected_reward).item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'dpo_loss: {current_loss:.4f}, reward_margin: {reward_margin:.4f}, '
                f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    'dpo_loss': current_loss,
                    'reward_margin': reward_margin,
                    'chosen_reward': chosen_reward.item(),
                    'rejected_reward': rejected_reward.item(),
                    'learning_rate': current_lr,
                    'eta': eta_min,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_train_state(epoch, step, wandb)

        del chosen_ids, chosen_labels, rejected_ids, rejected_labels, pixel_values
        del policy_chosen, policy_rejected, ref_chosen, ref_rejected, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V RLHF-V DPO (Qwen2.5-0.5B + SigLIP2)")
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='rl_vlm', type=str, help="Weight prefix")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Preference batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Mixed precision dtype")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loading threads")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip threshold")
    parser.add_argument("--log_interval", type=int, default=2, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Save interval")
    parser.add_argument('--hidden_size', default=896, type=int, help="Hidden size (Qwen2.5-0.5B=896)")
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="Number of layers (Qwen2.5-0.5B=24)")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Max sequence length")
    parser.add_argument("--data_path", type=str, default="../rl_dataset/RLHF-V/dataset_2/data.jsonl", help="RLHF-V data.jsonl path")
    parser.add_argument("--image_root", type=str, default=None, help="Image root directory (default: data_path parent)")
    parser.add_argument('--from_weight', default='sft_250k_10k', type=str, help="SFT weight prefix to start from")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto resume (0=no, 1=yes)")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="Policy freeze strategy")
    parser.add_argument('--max_samples', default=None, type=int, help="Limit training samples (None = all)")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Apply LoRA to Qwen attention layers")
    parser.add_argument("--average_log_prob", action="store_true", help="Use length-normalized sequence logprobs")
    parser.add_argument("--length_penalty", type=float, default=0.0, help="Length penalty per token (e.g. 0.01), use with --average_log_prob")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="Use torch.compile for policy model")
    parser.add_argument("--freeze_projector", action="store_true", help="Freeze vision projector, only train LoRA")
    parser.add_argument("--use_wandb", action="store_true", help="Use swanlab logging")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-DPO-Qwen", help="Wandb/SwanLab project name")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
    )
    ckp_data = vlm_checkpoint(
        vlm_config, weight=args.save_weight, save_dir='../checkpoints'
    ) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"MiniMind-V-DPO-Qwen-Epoch-{args.epochs}-BS-{args.batch_size}-"
            f"LR-{args.learning_rate}-Beta-{args.beta}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        freeze_projector=args.freeze_projector,
    )
    ref_model, _, _ = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=2,
        use_lora=False,
        freeze_projector=args.freeze_projector,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    train_ds = RLHFDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        max_length=vlm_config.max_position_embeddings,
        max_samples=args.max_samples,
        image_root=args.image_root,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled for policy model')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device_type == "cuda"),
            collate_fn=dpo_collate_fn,
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping first {start_step} steps')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
        start_step = 0

    if is_main_process():
        save_eval_weights()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
