import json
from pathlib import Path
from threading import Lock

CONFIG_PATH = Path(__file__).parent / "prompts.json"

DEFAULT_CONFIG = {
    "system_prompt": "你是一个资深的杂志编辑，擅长用简洁、优雅的中文回答问题。",
    "user_prompt_template": "{prompt}",
}


class PromptsConfig:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _load(self) -> dict:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return {**DEFAULT_CONFIG, **data}
        return dict(DEFAULT_CONFIG)

    def _save(self, data: dict):
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def get_system_prompt(self) -> str:
        return self._load().get("system_prompt", DEFAULT_CONFIG["system_prompt"])

    def get_user_prompt_template(self) -> str:
        return self._load().get("user_prompt_template", DEFAULT_CONFIG["user_prompt_template"])

    def get_all(self) -> dict:
        return self._load()

    def update(self, system_prompt: str | None = None, user_prompt_template: str | None = None) -> dict:
        with self._lock:
            current = self._load()
            if system_prompt is not None:
                current["system_prompt"] = system_prompt
            if user_prompt_template is not None:
                current["user_prompt_template"] = user_prompt_template
            self._save(current)
        return current
