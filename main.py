"""
main.py — the web layer.

Three endpoints:
  GET  /health   is the server alive and did the model load?
  POST /predict  just the CNN, no LLM. Send an image, get a disease name.
  POST /ask      the full agent. Send a question, optionally with an image.

Run locally:  uvicorn main:app --reload --port 7860
Then open:    http://localhost:7860
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

load_dotenv()  # reads .env when running locally; ignored in the cloud

import agent as brain

# Holds everything built at startup. A dict rather than globals so it is
# obvious where the state lives.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server boots, before it accepts any request.

    This is the whole reason a server is faster than a notebook: loading the
    CNN takes a few seconds, and we pay that cost once instead of per request.
    """
    print("Loading model and building agent...")
    try:
        state["agent"], state["detector"], state["classes"] = brain.build_agent()
        print("Ready. Classes:", state["classes"])
    except Exception as exc:  # noqa: BLE001 -- we genuinely want every failure
        # Do not let the process die here. If it exits, Render restarts it in a
        # loop and all you get is the same traceback scrolling past. Staying up
        # means /health can tell you in one line what went wrong -- nine times
        # out of ten a missing GOOGLE_API_KEY or a model file that never made
        # it into the image.
        state["error"] = f"{type(exc).__name__}: {exc}"
        print("STARTUP FAILED:", state["error"])
    yield
    state.clear()


app = FastAPI(
    title="Farmer Assistant API",
    description="Tomato leaf disease detection (CNN) plus grounded treatment advice (RAG).",
    version="1.0.0",
    lifespan=lifespan,
)


def save_upload(upload: UploadFile) -> str:
    """Write an uploaded file to a temp path and return that path.

    The CNN tool takes a file path, not bytes, so the upload has to touch disk.
    /tmp is the only reliably writable directory inside a container.
    """
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(400, "That file is not an image. Upload a JPG or PNG.")

    suffix = os.path.splitext(upload.filename or "leaf.jpg")[1] or ".jpg"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    with handle as out:
        shutil.copyfileobj(upload.file, out)
    return handle.name


def require_ready():
    """Guard the endpoints that cannot work until build_agent() has succeeded."""
    if "agent" not in state:
        raise HTTPException(503, state.get("error", "Model is still loading, try again shortly."))


@app.get("/health")
def health():
    """Cloud platforms poll this to decide if the container is healthy.

    Render treats 2xx as healthy, so a broken startup has to answer with 503.
    Returning 200 with model_loaded=false would leave a dead service marked live.
    """
    ready = "agent" in state
    body = {
        "status": "ok" if ready else "error",
        "model_loaded": ready,
        "classes": state.get("classes", []),
    }
    if not ready:
        body["error"] = state.get("error", "still starting")
        return JSONResponse(body, status_code=503)
    return body


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """CNN only. Useful for showing students the difference in speed and cost:
    this costs nothing and takes milliseconds, /ask calls an LLM several times."""
    require_ready()
    path = save_upload(image)
    try:
        return {"result": state["detector"].invoke(path)}
    finally:
        os.remove(path)


@app.post("/ask")
async def ask(question: str = Form(...), image: UploadFile | None = File(None)):
    """The full agent. If an image comes with the question, its path is appended
    to the text so the agent can decide to call the CNN tool."""
    require_ready()
    path = None
    try:
        if image is not None and image.filename:
            path = save_upload(image)
            question = f"{question}\n\nHere is a photo of the leaf: {path}"
        return brain.ask(state["agent"], question)
    finally:
        if path and os.path.exists(path):
            os.remove(path)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")
