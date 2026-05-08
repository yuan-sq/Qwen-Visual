from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from chat_manager import ChatManager
from model_loader import ModelLoader
from prompts_config import PromptsConfig

app = FastAPI(title="Vintage Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ChatManager()
loader = ModelLoader()
prompts_cfg = PromptsConfig()

WEB_APP_DIR = Path(__file__).parent.parent
INDEX_HTML = WEB_APP_DIR / "index.html"


@app.get("/")
def index():
    return FileResponse(str(INDEX_HTML))


class ChatRequest(BaseModel):
    session_id: str
    message: str
    image_base64: Optional[str] = None
    single_turn: bool = False


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class PromptsUpdateRequest(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None


class SwitchModelRequest(BaseModel):
    model_name: str


@app.on_event("startup")
def startup():
    loader.initialize()


@app.get("/sessions")
def list_sessions():
    return manager.list_sessions()


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    session = manager.create_session(title=req.title or "新对话")
    return session


@app.get("/chat/{session_id}")
def get_chat(session_id: str):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/prompts")
def get_prompts():
    return prompts_cfg.get_all()


@app.put("/prompts")
def update_prompts(req: PromptsUpdateRequest):
    return prompts_cfg.update(
        system_prompt=req.system_prompt,
        user_prompt_template=req.user_prompt_template,
    )


@app.get("/models")
def list_models():
    return loader.list_models()


@app.get("/models/current")
def current_model():
    return {"model": loader.get_current_model()}


@app.post("/models/switch")
def switch_model(req: SwitchModelRequest):
    return loader.switch_model(req.model_name)


@app.delete("/chat/{session_id}")
def delete_chat(session_id: str):
    success = manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.post("/chat")
def post_chat(req: ChatRequest):
    session = manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Append user message
    manager.append_message(req.session_id, "user", req.message, req.image_base64)

    # Stream chunks in real-time, then save the full response
    # When single_turn=True, omit history so model only sees current input
    history = [] if req.single_turn else session["messages"][:-1]
    chunks = []

    def event_stream():
        for chunk in loader.stream_chat(req.message, req.image_base64, history):
            chunks.append(chunk)
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

        # Save assistant response after streaming completes
        full_response = "".join(chunks)
        manager.append_message(req.session_id, "assistant", full_response)

    return StreamingResponse(event_stream(), media_type="text/event-stream")