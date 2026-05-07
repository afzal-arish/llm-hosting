from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model config from environment (fallback to Qwen 0.5B) ──────────────────
HF_REPO     = os.getenv("HF_REPO",     "Qwen/Qwen1.5-0.5B-Chat-GGUF")
HF_FILENAME = os.getenv("HF_FILENAME", "qwen1_5-0_5b-chat-q4_k_m.gguf")
HF_TOKEN    = os.getenv("HF_TOKEN",    None)   # only needed for private repos
MODEL_PATH  = os.getenv("MODEL_PATH",  "model.gguf")

# ── Global state ────────────────────────────────────────────────────────────
llm          = None
model_status = "loading"   # "loading" | "ready" | "error"
model_error  = ""


def download_and_load():
    """Download model from HuggingFace Hub (if not cached) then load it."""
    global llm, model_status, model_error

    try:
        # --- Download -------------------------------------------------------
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Downloading {HF_FILENAME} from {HF_REPO} ...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id   = HF_REPO,
                filename  = HF_FILENAME,
                token     = HF_TOKEN,
                local_dir = ".",
            )
            # hf_hub_download saves to a cache path; rename to MODEL_PATH
            if downloaded != MODEL_PATH:
                import shutil
                shutil.copy(downloaded, MODEL_PATH)
            logger.info("Download complete.")
        else:
            logger.info("Model file already present — skipping download.")

        # --- Load -----------------------------------------------------------
        logger.info("Loading model into memory ...")
        from llama_cpp import Llama
        llm = Llama(
            model_path = MODEL_PATH,
            n_ctx      = 512,      # reasonable context window
            n_threads  = 2,        # Render gives 2 vCPUs on free tier
            n_batch    = 64,
            verbose    = False,
        )
        model_status = "ready"
        logger.info("Model ready.")

    except Exception as e:
        model_status = "error"
        model_error  = str(e)
        logger.error(f"Model load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model download + load in a background thread so the
    # HTTP server comes up immediately (passes Render's health check).
    t = threading.Thread(target=download_and_load, daemon=True)
    t.start()
    yield
    # nothing special on shutdown


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "LLM Chat API",
    description = "Tiny LLM hosted on Render",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status:   str


# ── Prompt template ──────────────────────────────────────────────────────────
def build_prompt(msg: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant. Give short, clear answers.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{msg}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Render uses this for health checks."""
    return {"status": model_status, "error": model_error if model_error else None}


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Graceful fallback — pure JSON info so health check never 500s
        return HTMLResponse(
            content="<h2>LLM Chat API is starting up…</h2>"
                    "<p>Check <a href='/health'>/health</a> for model status.</p>"
                    "<p>Send POST to <code>/chat</code> with <code>{\"message\": \"hello\"}</code></p>",
            status_code=200,
        )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model_status == "loading":
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Try again in a moment.",
        )
    if model_status == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to load: {model_error}",
        )

    try:
        prompt = build_prompt(req.message)
        output = llm(
            prompt,
            max_tokens  = 256,
            temperature = 0.7,
            top_p       = 0.9,
            stop        = ["<|im_end|>", "<|im_start|>"],
            echo        = False,
        )
        text = output["choices"][0]["text"].strip()
        return ChatResponse(response=text or "No response generated.", status="ok")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))