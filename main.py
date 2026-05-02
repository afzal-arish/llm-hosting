from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
import os

app = FastAPI()

# ✅ Load model safely
MODEL_PATH = "model.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.gguf not found in project folder")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,
    n_threads=1,
    n_batch=32,
    verbose=False
)

# ✅ Request schema
class ChatRequest(BaseModel):
    message: str

# ✅ Prompt template (important for clean answers)
def build_prompt(msg):
    return f"""### Instruction:
Give a short and clear answer.

### User:
{msg}

### Response:
"""

# ✅ Serve frontend
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ✅ Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        prompt = build_prompt(req.message)

        output = llm(
            prompt,
            max_tokens=80,
            temperature=0.6,
            top_p=0.9,
            stop=["###"]
        )

        text = output["choices"][0]["text"].strip()

        return {"response": text if text else "No response generated."}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}