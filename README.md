# Qwen-Visual

Qwen-Visual 是一个轻量级视觉语言模型（VLM）实验项目。基于 Qwen2.5-0.5B-Instruct 语言模型和 SigLIP2-base-patch16-224 视觉编码器，通过 MLP 投影器将图像特征接入语言模型，完成了预训练、监督微调（SFT）、DPO 对齐训练以及 Web Demo。

## 模型架构

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  SigLIP2     │────▶│  MLP Projector  │────▶│  Qwen2.5-0.5B    │
│  (frozen)    │     │  (trainable)     │     │  + LoRA (SFT/DPO)│
│  768-d       │     │  768 → 896      │     │  896-d, 24 layers│
└──────────────┘     └─────────────────┘     └──────────────────┘
     ~95M                ~1.49M                  ~494M + ~2.6M LoRA
```

- **Vision Encoder**: SigLIP2-base-patch16-224（224×224 输入，196 patches，始终冻结）
- **Projector**: LayerNorm → Linear(768,896) → GELU → Linear(896,896)
- **LLM**: Qwen2.5-0.5B-Instruct（Pretrain 阶段冻结，SFT/DPO 阶段加 LoRA）

## 目录结构

```text
.
├── model/                    # QwenVL 模型定义
│   ├── model_vlm.py          #   VLMConfig, MMVisionProjector, QwenVL
│   └── __init__.py
├── trainer/                  # 训练脚本
│   ├── train_pretrain_vlm.py #   投影器预训练 (freeze_llm=2)
│   ├── train_sft_vlm.py      #   SFT LoRA 微调 (freeze_llm=1)
│   ├── train_rl_vlm.py       #   DPO 对齐训练
│   └── trainer_utils.py      #   模型初始化、checkpoint、数据工具
├── eval.py                   # 推理评测（单模型，eval_images/）
├── eval_images/              # 定性测试图片
├── build_rl_dataset/         # DPO 数据集构建工具
│   ├── sample.py             #   采样多响应
│   └── label_rl_dataset.py   #   LLM 标注中文 chosen/rejected
├── scripts/                  # 辅助脚本
│   ├── convert_vlm.py        #   .pth → transformers 格式转换
│   └── web_demo_vlm.py       #   Gradio Demo
├── web_app/                  # FastAPI Web 应用
│   ├── backend/              #   后端（模型加载、会话管理、SSE 流式）
│   └── index.html            #   前端页面
├── llm.py                    # LLM API 调用封装（DeepSeek）
└── requirements.txt          # Python 依赖
```

> 以下目录需自行准备，不入版本库：
> - `models/` — 基座模型（Qwen2.5-0.5B-Instruct、SigLIP2）
> - `out/` — 训练产出的权重文件
> - `dataset/` — 训练数据（Parquet/JSONL）

## 环境准备

Python 3.10+，先安装 PyTorch，再安装依赖：

```bash
# GPU (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# macOS / CPU
pip install torch torchvision

# 项目依赖
pip install -r requirements.txt
```

### 基座模型下载

```bash
mkdir -p models/Qwen models/google

# Qwen2.5-0.5B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct models/Qwen/Qwen2.5-0.5B-Instruct

# SigLIP2 视觉编码器
git clone https://huggingface.co/google/siglip2-base-patch16-224 models/google/siglip2-base-patch16-224
```

## 训练

### 1. 投影器预训练（可选）

冻结 Qwen + SigLIP2，只训练 projector：

```bash
cd trainer && python train_pretrain_vlm.py --epochs 2 --num_workers 0
```

输出：`out/pretrain_vlm_896.pth`

### 2. SFT 微调（必须）

冻结 SigLIP2，训练 projector + Qwen LoRA（r=8，全部 24 层 attention）：

```bash
cd trainer && python train_sft_vlm.py --epochs 3 --from_weight pretrain_vlm --num_workers 0

# 或跳过 VLM 预训练，直接从 Qwen 原始权重开始
cd trainer && python train_sft_vlm.py --epochs 3 --from_weight none --num_workers 0
```

输出：`out/sft_vlm_896.pth`

### 3. DPO 对齐训练（可选）

加载 SFT checkpoint 作为策略模型 + 冻结参考模型：

```bash
cd trainer && python train_rl_vlm.py --from_weight sft_250k_10k --epochs 1 --beta 0.1 --num_workers 0
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--from_weight` | 基础 VLM checkpoint 前缀（`pretrain_vlm`/`sft_vlm`），`none` 为从 Qwen 原始权重开始 |
| `--freeze_llm` | 2=仅 projector, 1=projector+LoRA, 0=全量微调 |
| `--max_samples` | 限制数据量，用于快速测试 |
| `--from_resume 1` | 断点续训 |
| `--use_wandb` | 启用 SwanLab 日志 |
| `--num_workers 0` | **必须为 0**，否则 SkipBatchSampler 会死锁 |

### DPO 数据构建

```bash
# 1. 对每张图采样多个响应
python build_rl_dataset/sample.py --dataset_dir rl_dataset/RLHF-V/dataset_2 --num_samples 4

# 2. 用 LLM 标注中文 chosen/rejected（需设置 DEEPSEEK_API_KEY 环境变量）
python build_rl_dataset/label_rl_dataset.py --dataset_dir rl_dataset/RLHF-V/dataset_1
```

## 推理评测

```bash
# 对 eval_images/ 中的图片逐张生成描述
python eval.py --load_from model --save_dir out --weight sft_vlm --device cuda

# 加载 transformers 格式模型
python eval.py --load_from ./minimind-3v --device cuda

# CPU 推理（速度较慢）
python eval.py --load_from model --device cpu
```

## Web Demo

### FastAPI 后端

```bash
cd web_app/backend && uvicorn main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000
```

功能：图片上传、多轮视觉问答、SSE 流式输出、会话管理、动态 prompt 配置

### Gradio Demo

```bash
cd scripts && python web_demo_vlm.py
```

## 模型格式转换

将原生 `.pth` 权重转换为 HuggingFace transformers 格式：

```bash
cd scripts && python convert_vlm.py
```

## 已知问题

- **`--num_workers` 死锁**: 使用 SkipBatchSampler 时 `num_workers > 0` 会在 ~6000 步死锁，必须设为 0
- **训练脚本工作目录**: 必须从 `trainer/` 目录运行（`cd trainer && python train_*.py`）
- **LoRA 合并**: SFT checkpoint 保存前会自动调用 `merge_and_unload()`，无需手动处理
- **transformers 5.x**: 加载本地模型时需使用 `os.path.abspath()` + `local_files_only=True`

## 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — 项目灵感来源

