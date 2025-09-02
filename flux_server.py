#!/usr/bin/env python3
import os
import json
import argparse
import shutil
from datetime import datetime
from threading import Timer
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from diffusers import FluxPipeline, StableDiffusionPipeline

# -------------------
# CLI args
# -------------------
parser = argparse.ArgumentParser(description="FLUX/SD server")
parser.add_argument("--config", default="config.json", help="Path to config.json")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# -------------------
# Load config
# -------------------
with open(args.config) as f:
    config = json.load(f)

DEFAULT_MODEL: str = config["default_model"]
UNLOAD_TIMEOUT: int = int(config.get("unload_timeout", 300))
MEMORY_FRACTION: float = float(config.get("memory_fraction", 0.8))
DEFAULTS = config.get("defaults", {})
AVAILABLE_MODELS: List[str] = list(config.get("available_models", []))

HF_HUB_BASE = os.path.expanduser("~/.cache/huggingface/hub")
HF_DIFFUSERS_BASE = os.path.expanduser("~/.cache/huggingface/diffusers")

# -------------------
# FastAPI + Globals
# -------------------
app = FastAPI(title="Flux Server", version="1.0")
pipe = None
current_model: Optional[str] = None
unload_timer: Optional[Timer] = None

# Reserve VRAM headroom for desktop/compositor
if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    except Exception:
        pass  # best-effort

# -------------------
# Models
# -------------------
class GenRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None
    num_images: Optional[int] = None
    seed: Optional[int] = None
    outfile: Optional[str] = "out.png"
    preview: Optional[bool] = False  # not used server-side, but kept for symmetry


def _discover_cached_models() -> List[str]:
    def scan(base: str) -> List[str]:
        found = set()
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.startswith("models--") and os.path.isdir(os.path.join(base, name)):
                    rid = name[len("models--"):].replace("--", "/")
                    found.add(rid)
        return sorted(found)

    cached = set(scan(HF_HUB_BASE)) | set(scan(HF_DIFFUSERS_BASE))
    # include user-declared available_models too
    cached |= set(AVAILABLE_MODELS)
    return sorted(cached)


def _repo_dir(base: str, model_id: str) -> str:
    return os.path.join(base, "models--" + model_id.replace("/", "--"))


def _remove_model_from_cache(model_id: str) -> List[str]:
    removed = []
    for base in (HF_HUB_BASE, HF_DIFFUSERS_BASE):
        path = _repo_dir(base, model_id)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


def _unload():
    global pipe, current_model, unload_timer
    if pipe is not None:
        print(f"💤 Unloading model: {current_model}")
        del pipe
        pipe = None
        current_model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if unload_timer:
        unload_timer.cancel()
        unload_timer = None


def _reset_unload_timer():
    global unload_timer
    if unload_timer:
        unload_timer.cancel()
    unload_timer = Timer(UNLOAD_TIMEOUT, _unload)
    unload_timer.start()


def _load(model_id: str):
    global pipe, current_model
    _unload()
    print(f"🚀 Loading model: {model_id}")

    # Choose pipeline
    if "flux" in model_id.lower():
        pipe_ = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    else:
        pipe_ = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Memory-friendly toggles
    pipe_.enable_attention_slicing()
    # NOTE: only one offload method at a time; model_cpu_offload is a good default
    pipe_.enable_model_cpu_offload()

    current_model = model_id
    return pipe_


# -------------------
# API Endpoints
# -------------------
@app.get("/health")
def health():
    return {"status": "ok", "loaded": current_model}


@app.get("/defaults")
def get_defaults():
    return {
        "default_model": DEFAULT_MODEL,
        "unload_timeout": UNLOAD_TIMEOUT,
        "memory_fraction": MEMORY_FRACTION,
        "defaults": DEFAULTS
    }


@app.get("/models")
def list_models():
    return {"available": _discover_cached_models(), "loaded": current_model}


@app.post("/load")
def load_model(model: str = Query(..., description="Model repo id, e.g. black-forest-labs/FLUX.1-schnell")):
    global pipe
    pipe = _load(model)
    return {"status": "loaded", "model": model}


@app.post("/unload")
def unload_model():
    _unload()
    return {"status": "unloaded"}


@app.delete("/models")
def delete_model(model: str = Query(..., description="Model repo id to remove from cache")):
    if current_model == model:
        _unload()
    removed_paths = _remove_model_from_cache(model)
    if not removed_paths:
        raise HTTPException(status_code=404, detail="Model not found in cache")
    return {"status": "removed", "model": model, "paths": removed_paths}


@app.post("/generate")
def generate(req: GenRequest):
    global pipe

    model_id = req.model or DEFAULT_MODEL
    if current_model != model_id:
        pipe = _load(model_id)

    steps = req.steps or DEFAULTS.get("steps", 6)
    width = req.width or DEFAULTS.get("width", 512)
    height = req.height or DEFAULTS.get("height", 512)
    guidance = req.guidance or DEFAULTS.get("guidance", 3.5)
    num_images = req.num_images or DEFAULTS.get("num_images", 1)
    seed = req.seed if req.seed is not None else DEFAULTS.get("seed", None)
    outfile = req.outfile or "out.png"

    # Optional: set seed
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    else:
        generator = None

    print(f"🎨 [{current_model}] steps={steps} size={width}x{height} gs={guidance} n={num_images} seed={seed}")
    result = pipe(
        req.prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        generator=generator
    ).images

    # Save one or many
    saved = []
    if num_images == 1:
        path = outfile
        result[0].save(path)
        saved.append(path)
    else:
        root, ext = os.path.splitext(outfile)
        for i, img in enumerate(result):
            path = f"{root}_{i+1}{ext or '.png'}"
            img.save(path)
            saved.append(path)

    _reset_unload_timer()
    return {
        "status": "ok",
        "model": current_model,
        "saved": saved
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

