import sys
import torch
import warnings
import base64
import io
import re
from pathlib import Path
from threading import Lock
from queue import Queue
from threading import Thread

# Ensure project root is on sys.path so 'model' module can be found
_proj_root = Path(__file__).parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent.parent.parent

QWEN_PATH = BASE_DIR / "models" / "Qwen" / "Qwen2.5-0.5B-Instruct"
SIGLIP_PATH = BASE_DIR / "models" / "google" / "siglip2-base-patch16-224"
OUT_DIR = BASE_DIR / "out"

IMAGE_PAD_TOKEN = "<|image_pad|>"


class ModelLoader:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._current_name = None
            cls._instance._model_cache = {}
        return cls._instance

    def list_models(self):
        """Scan out/ directory for available .pth model files."""
        models = []
        if OUT_DIR.exists():
            for p in sorted(OUT_DIR.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True):
                name = p.stem  # e.g. "sft_250k_10k_896" or "rl_5k_896"
                # Strip the _896 suffix for display if present
                display = re.sub(r'_896$', '', name)
                models.append({"name": name, "display": display, "path": str(p)})
        return models

    def switch_model(self, model_name: str):
        """Switch to a different model by name (without .pth extension)."""
        model_path = OUT_DIR / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_name == self._current_name and self._current_name in self._model_cache:
            print(f"[ModelLoader] Already using {model_name}, no reload needed")
            return {"ok": True, "model": model_name}

        # Cache current model before switching
        if self._current_name and self._current_name not in self._model_cache:
            self._model_cache[self._current_name] = self.model

        # Load from cache if available
        if model_name in self._model_cache:
            print(f"[ModelLoader] Switching to cached model: {model_name}")
            self.model = self._model_cache[model_name]
            self._current_name = model_name
            return {"ok": True, "model": model_name}

        print(f"[ModelLoader] Loading model: {model_name} from {model_path}")
        print(f"[ModelLoader] Device: {DEVICE}")

        from model.model_vlm import QwenVL, VLMConfig

        lm_config = VLMConfig(hidden_size=896, num_hidden_layers=24, vocab_size=len(self.tokenizer))
        lm_config.image_special_token = IMAGE_PAD_TOKEN
        img_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        lm_config.image_ids = [img_id]
        lm_config.image_token_len = 196

        model = QwenVL(lm_config, vision_model_path=str(SIGLIP_PATH))
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
        del state_dict

        model = model.half().eval().to(DEVICE)
        model.vision_encoder = model.vision_encoder.to(DEVICE)

        self.model = model
        self._model_cache[model_name] = model
        self._current_name = model_name
        print(f"[ModelLoader] Model {model_name} loaded successfully")
        return {"ok": True, "model": model_name}

    def get_current_model(self):
        return self._current_name

    def initialize(self):
        if self._initialized:
            return

        print(f"[ModelLoader] Initializing tokenizer and base model")
        print(f"[ModelLoader] Device: {DEVICE}")

        from transformers import AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(QWEN_PATH), trust_remote_code=True
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PAD_TOKEN]})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._initialized = True

        # Auto-load the first available model
        available = self.list_models()
        if available:
            first = available[0]["name"]
            self.switch_model(first)
        else:
            print("[ModelLoader] Warning: no .pth files found in out/")

    def chat(self, prompt: str, image_base64: str | None = None, history: list[dict] | None = None):
        from PIL import Image
        from transformers import TextStreamer
        from prompts_config import PromptsConfig

        if not self._initialized:
            self.initialize()

        if history is None:
            history = []

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        # Prepend system prompt if not already present
        prompts_cfg = PromptsConfig()
        system_prompt = prompts_cfg.get_system_prompt()
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Build current user message with image placeholders if needed
        user_prompt_template = prompts_cfg.get_user_prompt_template()
        user_content = user_prompt_template.replace("{prompt}", prompt) if "{prompt}" in user_prompt_template else f"{user_prompt_template}{prompt}"
        if image_base64:
            placeholder = IMAGE_PAD_TOKEN * 196
            user_content = f"{placeholder}\n{user_content}"
        messages.append({"role": "user", "content": user_content})

        pixel_values = None
        if image_base64:
            image_data = base64.b64decode(image_base64.split(",")[-1])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            from model.model_vlm import QwenVL as VLMClass
            pixel_values = {k: v.to(DEVICE) for k, v in VLMClass.image2tensor(image, self.model.processor).items()}

        new_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )[-8192 + 1:]

        inputs = self.tokenizer(new_prompt, return_tensors="pt", truncation=True).to(DEVICE)

        queue = Queue()

        class AsyncStreamer(TextStreamer):
            def __init__(self, tokenizer, queue):
                super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                self.queue = queue

            def on_finalized_text(self, text: str, stream_end: bool = False):
                self.queue.put(text)
                if stream_end:
                    self.queue.put(None)

        streamer = AsyncStreamer(self.tokenizer, queue)

        def _generate():
            with self._lock:
                self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    pixel_values=pixel_values,
                )

        Thread(target=_generate, daemon=True).start()

        while True:
            text = queue.get()
            if text is None:
                break
            yield text

    def stream_chat(self, prompt: str, image_base64: str | None = None, history: list[dict] | None = None):
        yield from self.chat(prompt, image_base64, history)