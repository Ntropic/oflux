#!/usr/bin/env python3
import os
# Reduce frag; note ROCm prints a warning but it's harmless.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import json
import argparse
import shutil
from threading import Timer
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from diffusers import FluxPipeline, StableDiffusionPipeline
import base64
from io import BytesIO
import re
from datetime import datetime

STOPWORDS = {"a", "as", "from", "to", "with", "the", "in", "of", "on", "for", "and"}

def summarize_prompt(prompt: str, max_words: int = 6) -> str:
    """Summarize prompt into a safe filename prefix with underscores."""
    words = [
        re.sub(r"\W+", "", w.lower())  # clean non-alphanumeric
        for w in prompt.split()
        if w.lower() not in STOPWORDS
    ]
    # filter empty and cut length
    words = [w for w in words if w][:max_words]
    prefix = "_".join(words)
    return prefix or datetime.now().strftime("gen_%Y%m%d_%H%M%S")

# ---- CLI ----
parser = argparse.ArgumentParser(description="FLUX/SD server")
parser.add_argument("--config", default="config.json", help="Path to config.json")
parser.add_argument("--models", default="models.json", help="Path to models.json (managed list)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---- config + models ----
def load_json(path, default):
    if not os.path.exists(path):
        with open(path, "w") as f: json.dump(default, f, indent=2)
        return default.copy()
    with open(path) as f: return json.load(f)

config = load_json(args.config, {
    "unload_timeout": 300,
    "memory_fraction": 0.75,
    "offload": "sequential",
    "defaults": {"steps": 4, "width": 512, "height": 512, "guidance": 3.5, "num_images": 1, "seed": None, "dtype": "bfloat16"},
    "default_model": "black-forest-labs/FLUX.1-schnell",
    "default_lora": "ostris/flux-dev-lora",
})
models_state = load_json(args.models, {"models": [], "loaded": None, "loras": [], "last": None})
models_state.setdefault("loras", [])
models_state.setdefault("last", None)

UNLOAD_TIMEOUT: int = int(config.get("unload_timeout", 300))
MEMORY_FRACTION: float = float(config.get("memory_fraction", 0.75))
OFFLOAD_KIND: str = str(config.get("offload", "sequential")).lower()
DEFAULTS = config.get("defaults", {})
DEFAULT_MODEL: str = config.get("default_model", "")
DEFAULT_LORA: Optional[str] = config.get("default_lora")

HF_HUB_BASE = os.path.expanduser("~/.cache/huggingface/hub")
HF_DIFFUSERS_BASE = os.path.expanduser("~/.cache/huggingface/diffusers")

def save_models():
    with open(args.models, "w") as f: json.dump(models_state, f, indent=2)

# ---- app state ----
app = FastAPI(title="Flux Server", version="1.3")
pipe = None
current_model: Optional[str] = None
unload_timer: Optional[Timer] = None
current_lora: Optional[str] = None
current_lora_scale: float = 1.0

# Reserve VRAM headroom, best-effort
if torch.cuda.is_available():
    try: torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    except Exception: pass

# ---- helpers ----
def _remove_model_from_cache(model_id: str) -> List[str]:
    def path_for(base: str, model: str) -> str:
        return os.path.join(base, "models--" + model.replace("/", "--"))
    removed = []
    for base in (HF_HUB_BASE, HF_DIFFUSERS_BASE):
        p = path_for(base, model_id)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
    return removed

def _unload():
    global pipe, current_model, unload_timer, current_lora, current_lora_scale
    if pipe is not None:
        print(f"[Unloading] Unloading model: {current_model}")
        del pipe
        pipe = None
        current_model = None
        current_lora = None
        current_lora_scale = 1.0
        try: torch.cuda.empty_cache()
        except Exception: pass
    models_state["loaded"] = None
    save_models()
    if unload_timer:
        unload_timer.cancel()
        unload_timer = None

def _reset_unload_timer():
    global unload_timer
    if unload_timer: unload_timer.cancel()
    unload_timer = Timer(UNLOAD_TIMEOUT, _unload)
    unload_timer.start()

def _choose_dtype(name: Optional[str]):
    n = (name or "").lower()
    if n in ("bf16", "bfloat16"): return torch.bfloat16
    if n in ("fp16", "float16", "half"): return torch.float16
    return torch.float32

def _apply_offload(p):
    # choose ONE: sequential (aggressive, low VRAM) OR model (lighter)
    p.enable_attention_slicing()
    if OFFLOAD_KIND == "sequential":
        p.enable_sequential_cpu_offload()
    elif OFFLOAD_KIND == "model":
        p.enable_model_cpu_offload()
    # else "none": no offload

def _load(model_id: str):
    """Load a model; on success add to models.json."""
    global pipe, current_model, current_lora, current_lora_scale
    _unload()
    print(f"[Loading] Loading model: {model_id}")

    torch_dtype = _choose_dtype(DEFAULTS.get("dtype", "bfloat16"))

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )

    # ✅ match what worked in IPython
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()

    current_model = model_id
    models_state["loaded"] = model_id
    current_lora = None
    current_lora_scale = 1.0

    if model_id not in models_state["models"]:
        models_state["models"].append(model_id)
    save_models()

    return pipe

class GenRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None
    num_images: Optional[int] = None
    seed: Optional[int] = None
    lora: Optional[str] = None
    lora_scale: Optional[float] = None

    # Server saving is OPT-IN via outfile; default None means: don't save on server
    outfile: Optional[str] = None

    # Encoding the server uses if it returns bytes (and to suggest extensions)
    format: Optional[str] = "png"       # "png" | "jpg" | "webp"

    # Should the server return images as base64 so the client can save locally?
    want_bytes: Optional[bool] = True

# ---- endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "loaded": models_state["loaded"]}

@app.get("/defaults")
def get_defaults():
    return {
        "unload_timeout": UNLOAD_TIMEOUT,
        "memory_fraction": MEMORY_FRACTION,
        "offload": OFFLOAD_KIND,
        "defaults": DEFAULTS,
        "default_model": DEFAULT_MODEL,
        "default_lora": DEFAULT_LORA
    }

@app.get("/models")
def list_models():
    # Only the curated list
    return {
        "available": models_state["models"],
        "loaded": models_state["loaded"],
        "loras": models_state["loras"],
        "active_lora": current_lora,
        "active_lora_scale": current_lora_scale if current_lora else None,
    }


@app.get("/last")
def get_last():
    return {
        "last": models_state.get("last"),
        "loaded": models_state.get("loaded"),
        "active_lora": current_lora,
        "default_model": DEFAULT_MODEL,
        "default_lora": DEFAULT_LORA,
    }

@app.post("/load")
def load_model(model: str = Query(..., description="Repo id, e.g. black-forest-labs/FLUX.1-schnell")):
    global pipe
    try:
        pipe = _load(model)
        return {"status": "loaded", "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {model}: {e}")

@app.post("/unload")
def unload_model():
    _unload()
    return {"status": "unloaded"}

@app.delete("/models")
def delete_model(model: str = Query(..., description="Repo id to remove from list and cache")):
    if models_state["loaded"] == model:
        _unload()
    removed_paths = _remove_model_from_cache(model)
    if model in models_state["models"]:
        models_state["models"].remove(model)
        save_models()
    return {"status": "removed", "model": model, "paths": removed_paths}


@app.delete("/loras")
def delete_lora(lora: str = Query(..., description="LoRA repo id to remove from cache")):
    global current_lora, current_lora_scale
    if current_lora == lora:
        if pipe is not None:
            _apply_lora(None, None)
        else:
            current_lora = None
            current_lora_scale = 1.0
    removed_paths = _remove_model_from_cache(lora)
    if lora in models_state["loras"]:
        models_state["loras"].remove(lora)
        save_models()
    return {"status": "removed", "lora": lora, "paths": removed_paths}

def _clamp_and_seed(req):
    steps = int(req.steps or DEFAULTS.get("steps", 4))
    width = int(req.width or DEFAULTS.get("width", 512))
    height = int(req.height or DEFAULTS.get("height", 512))
    guidance = float(req.guidance or DEFAULTS.get("guidance", 3.5))
    num_images = int(req.num_images or DEFAULTS.get("num_images", 1))
    seed = req.seed if req.seed is not None else DEFAULTS.get("seed", None)

    width = max(64, min(width, 1024))
    height = max(64, min(height, 1024))
    gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    return steps, width, height, guidance, num_images, gen

def _apply_lora(adapter: Optional[str], scale: Optional[float]):
    global current_lora, current_lora_scale, pipe
    target_id = adapter or None
    target_scale = float(scale) if scale is not None else 1.0

    if pipe is None:
        if not target_id:
            current_lora = None
            current_lora_scale = 1.0
            return
        raise HTTPException(status_code=400, detail="Load a base model before applying a LoRA adapter")

    if target_id and not hasattr(pipe, "load_lora_weights"):
        raise HTTPException(status_code=400, detail="Loaded pipeline does not support LoRA adapters")

    # Nothing requested -> ensure we are clean
    if not target_id:
        if current_lora:
            try:
                if hasattr(pipe, "unload_lora_weights"):
                    pipe.unload_lora_weights()
            except Exception:
                pass
            try:
                if hasattr(pipe, "unfuse_lora"):
                    pipe.unfuse_lora()
            except Exception:
                pass
            current_lora = None
            current_lora_scale = 1.0
        return

    if current_lora == target_id and abs(current_lora_scale - target_scale) < 1e-6:
        return

    # Clear any previously applied adapter
    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    except Exception:
        pass
    try:
        if hasattr(pipe, "unfuse_lora"):
            pipe.unfuse_lora()
    except Exception:
        pass

    adapter_name = "oflux_lora"
    try:
        pipe.load_lora_weights(target_id, adapter_name=adapter_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LoRA {target_id}: {e}") from e

    try:
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([adapter_name], weights=[target_scale])
        elif hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=target_scale)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply LoRA {target_id}: {e}") from e

    current_lora = target_id
    current_lora_scale = target_scale
    if target_id not in models_state["loras"]:
        models_state["loras"].append(target_id)
        save_models()

def _try_generate(req, steps, width, height, guidance, num_images, generator):
    images = []
    i = 0
    while i < num_images:
        # how many to generate in this batch (max 4 because pipeline caps there)
        k = min(4, num_images - i)

        if req.seed is not None:
            # reproducible generators, one per image
            gens = [
                torch.Generator(device="cpu").manual_seed(int(req.seed) + j)
                for j in range(i, i + k)
            ]
        else:
            # non-deterministic generators, one per image
            gens = [
                torch.Generator(device="cpu").manual_seed(torch.seed())
                for _ in range(k)
            ] if generator is None else [generator for _ in range(k)]

        # run one batch
        out = pipe(
            req.prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            num_images_per_prompt=k,
            generator=gens
        )

        images.extend(out.images)
        i += k

    return images


@app.post("/generate")
def generate(req: GenRequest):
    global pipe
    model_id = req.model or (models_state["loaded"] or DEFAULT_MODEL)

    # Load (and add to list) if needed
    if current_model != model_id or pipe is None:
        try:
            pipe = _load(model_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load {model_id}: {e}")

    _apply_lora(req.lora, req.lora_scale)

    steps, width, height, guidance, num_images, generator = _clamp_and_seed(req)
    print(f"[Generating] [{current_model}] {width}x{height} steps={steps} gs={guidance} n={num_images} seed={req.seed}")

    # One-shot OOM retry: halve size & steps, force n=1
    try:
        images = _try_generate(req, steps, width, height, guidance, num_images, generator)
    except torch.OutOfMemoryError:
        try:
            w2, h2 = max(64, width // 2), max(64, height // 2)
            s2 = max(2, steps // 2)
            print(f"⚠️  OOM -> retry with {w2}x{h2}, steps={s2}, n=1")
            images = _try_generate(req, s2, w2, h2, guidance, 1, generator)
            width, height, steps, num_images = w2, h2, s2, 1
        except torch.OutOfMemoryError as e:
            raise HTTPException(
                status_code=507,
                detail=("Out of VRAM. Try smaller width/height or steps, "
                        "or lower 'memory_fraction' in config.json, "
                        "and keep offload='sequential'.")
            ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

    # ----- Save or return bytes -----
    saved = []          # absolute paths saved on server (if outfile is given)
    suggested = []      # filenames suggested for client
    images_b64 = []     # base64-encoded images (if want_bytes=True)
    seeds = []          # actual seeds used

    fmt = (req.format or "png").lower()
    ext = ".png" if fmt == "png" else ".jpg" if fmt in ("jpg", "jpeg") else ".webp"

    # Base root (explicit or from prompt)
    if req.outfile:
        root, ext_out = os.path.splitext(req.outfile)
        if ext_out:
            ext = ext_out
    else:
        root = summarize_prompt(req.prompt)

    # Per-image saving
    base_seed = req.seed if req.seed is not None else torch.seed() % (2**31)
    for i, img in enumerate(images):
        seed_i = base_seed + i
        seeds.append(seed_i)

        fname = f"{root}_seed{seed_i}{ext}"
        if req.outfile:  # save on server
            img.save(fname)
            saved.append(os.path.abspath(fname))

        suggested.append(os.path.basename(fname))

        if req.want_bytes:
            buf = BytesIO()
            pil_fmt = "PNG" if ext == ".png" else "JPEG" if ext in (".jpg", ".jpeg") else "WEBP"
            img.save(buf, format=pil_fmt)
            images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    models_state["last"] = {
        "prompt": req.prompt,
        "model": current_model,
        "lora": current_lora,
        "lora_scale": current_lora_scale if current_lora else None,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance": guidance,
        "num_images": len(images),
        "seed": req.seed,
        "format": fmt,
        "outfile": req.outfile,
        "want_bytes": bool(req.want_bytes),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_models()

    _reset_unload_timer()
    return {
        "status": "ok",
        "model": current_model,
        "lora": current_lora,
        "lora_scale": current_lora_scale if current_lora else None,
        "saved": saved,
        "files": suggested,
        "images_b64": images_b64,
        "seeds": seeds,   # 👈 added
        "width": width,
        "height": height,
        "steps": steps,
        "num_images": len(images)
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
