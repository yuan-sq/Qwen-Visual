import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

CHATS_DIR = Path(__file__).parent / "chats"
CHATS_DIR.mkdir(exist_ok=True)


class ChatManager:
    def create_session(self, title: str = "新对话") -> dict:
        session_id = uuid.uuid4().hex[:12]
        session = {
            "session_id": session_id,
            "title": title,
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(session)
        return session

    def get_session(self, session_id: str) -> dict | None:
        path = CHATS_DIR / f"chat_{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> list[dict]:
        sessions = []
        for path in CHATS_DIR.glob("chat_*.json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "title": data["title"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data.get("messages", [])),
                })
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def append_message(self, session_id: str, role: str, content: str, image: str | None = None):
        session = self.get_session(session_id)
        if not session:
            return None
        msg = {"role": role, "content": content}
        if image:
            msg["image"] = image
        session["messages"].append(msg)
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        path = CHATS_DIR / f"chat_{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def _save(self, session: dict):
        path = CHATS_DIR / f"chat_{session['session_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)