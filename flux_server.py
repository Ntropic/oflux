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

from diffusers import FluxPipeline, Flux2Pipeline, StableDiffusionPipeline
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import HfFolder, login, snapshot_download
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
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
CONFIG_PATH = args.config

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
    "model_defaults": {},
})
models_state = load_json(args.models, {"models": [], "loaded": None})

UNLOAD_TIMEOUT: int = int(config.get("unload_timeout", 300))
MEMORY_FRACTION: float = float(config.get("memory_fraction", 0.75))
OFFLOAD_KIND: str = str(config.get("offload", "sequential")).lower()
DEFAULTS = config.get("defaults", {})
DEFAULT_MODEL: str = config.get("default_model", "")
MODEL_DEFAULTS = config.setdefault("model_defaults", {})

HF_HUB_BASE = os.path.expanduser("~/.cache/huggingface/hub")
HF_DIFFUSERS_BASE = os.path.expanduser("~/.cache/huggingface/diffusers")

def save_models():
    with open(args.models, "w") as f: json.dump(models_state, f, indent=2)

def save_config():
    config["model_defaults"] = MODEL_DEFAULTS
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _initialize_state():
    """Normalize persisted state on import/startup before serving."""
    global MODEL_DEFAULTS

    # The on-disk "loaded" entry can only reflect a previous session; clear it
    # so the configured default (or explicit request) is always used when the
    # server starts fresh.
    if models_state.get("loaded"):
        print("[Startup] Clearing stale 'loaded' state; no model is loaded yet.")
        models_state["loaded"] = None
        save_models()

    if not isinstance(MODEL_DEFAULTS, dict):
        # Recover gracefully if the config was manually corrupted.
        MODEL_DEFAULTS = {}
        config["model_defaults"] = MODEL_DEFAULTS
        save_config()


_initialize_state()

# ---- app state ----
app = FastAPI(title="Flux Server", version="1.2")
pipe = None
current_model: Optional[str] = None
unload_timer: Optional[Timer] = None

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


def _candidate_model_ids(model_id: str):
    yield model_id
    prefix = "black-forest-labs/"
    if not model_id.lower().startswith(prefix):
        base = model_id.split("/")[-1]
        yield prefix + base

def _ensure_model_defaults(model_id: str) -> dict:
    if model_id not in MODEL_DEFAULTS:
        defaults = {}
        # Preserve the previous automatic bnb4 preference for Flux 2 models
        if "flux.2" in model_id.lower():
            defaults["quantize"] = "bnb4"
        MODEL_DEFAULTS[model_id] = defaults
        save_config()
    return MODEL_DEFAULTS[model_id]


def _merged_defaults_for(model_id: str) -> dict:
    per_model = _ensure_model_defaults(model_id)
    merged = DEFAULTS.copy()
    merged.update({k: v for k, v in per_model.items() if v is not None})
    return merged


def _resolve_model(requested: Optional[str]) -> str:
    """Pick a model for a request, favoring explicit choice or the configured default."""

    if requested:
        return requested

    # If a pipeline is already loaded, stick with it unless the caller overrides.
    if pipe is not None and current_model:
        return current_model

    # Fresh server start: ignore any stale models_state["loaded"] and rely on config.
    if DEFAULT_MODEL:
        return DEFAULT_MODEL

    # As a last resort, fall back to the first tracked model (if any).
    if models_state.get("models"):
        return models_state["models"][0]

    raise HTTPException(status_code=400, detail="No model specified and no default_model configured")


def _update_model_defaults(model_id: str, updates: dict) -> dict:
    md = _ensure_model_defaults(model_id)
    changed = False
    for key, val in updates.items():
        if key == "quantize" and (val or "").lower() == "auto":
            if "quantize" in md:
                md.pop("quantize", None)
                changed = True
            continue
        md[key] = val
        changed = True
    if changed:
        save_config()
    return md

def _unload():
    global pipe, current_model, unload_timer
    if pipe is not None:
        print(f"[Unloading] Unloading model: {current_model}")
        del pipe
        pipe = None
        current_model = None
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

def _select_pipeline_cls(model_id: str):
    mid = model_id.lower()
    # FLUX.2
    if "flux.2" in mid and Flux2Pipeline is not None:
        return Flux2Pipeline
    # FLUX.1 and other Flux variants
    if "flux.1" in mid or "flux" in mid:
        return FluxPipeline
    # Fallback - Stable Diffusion
    return StableDiffusionPipeline


def _quant_mode(model_id: str, override: Optional[str] = None) -> Optional[str]:
    pref = (override or "").lower() or None
    if pref is None:
        pref = (_ensure_model_defaults(model_id).get("quantize") or "").lower() or None
    if pref in (None, "", "none", "auto"):
        return None
    return pref


def _quant_config(pref: Optional[str]):
    if pref is None:
        return None
    if BitsAndBytesConfig is None:
        print("[Quantization] transformers/bitsandbytes not available; loading without quantization.")
        return None
    if pref in ("bnb4", "4bit", "bitsandbytes4", "bnb-4bit"):
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if pref in ("bnb8", "8bit", "bitsandbytes8", "bnb-8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    print(f"[Quantization] Unknown quantization preference '{pref}', skipping quantization.")
    return None


def _quantization_for(model_id: str, quantize: Optional[str] = None):
    """Return a BitsAndBytesConfig based on overrides or stored defaults."""
    pref = _quant_mode(model_id, override=quantize)
    return _quant_config(pref)

def _env_hf_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    return None

def _ensure_hf_auth(model_id: str) -> Optional[str]:
    """Ensure a HuggingFace token is available, prompting the user if needed."""

    # Already logged in (cached) or provided via env
    token = HfFolder.get_token() or _env_hf_token()
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            return HfFolder.get_token() or token
        except Exception as e:
            print(f"[Auth] Failed to use existing token: {e}")

    # Interactive login fallback
    print(
        f"[Auth] Model '{model_id}' requires authentication. "
        "Log in with your HuggingFace token to continue."
    )
    try:
        login(add_to_git_credential=False)
        return HfFolder.get_token()
    except Exception as e:
        print(f"[Auth] Login failed: {e}")
        return None


def _pull_model(model_id: str) -> str:
    """Download model files into the local HuggingFace cache without loading."""

    def _download(repo_id: str, token: Optional[str]):
        return snapshot_download(
            repo_id=repo_id,
            token=token,
            resume_download=True,
        )

    last_error = None
    for candidate in _candidate_model_ids(model_id):
        try:
            cache_dir = _download(candidate, HfFolder.get_token() or _env_hf_token())
        except HfHubHTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code in (401, 403):
                print(f"[Auth] Access to {candidate} requires authentication.")
                token = _ensure_hf_auth(candidate)
                if not token:
                    continue
                try:
                    cache_dir = _download(candidate, token)
                except Exception:
                    continue
            elif e.response is not None and e.response.status_code == 404:
                if candidate != model_id:
                    print(f"[Lookup] {model_id} not found; retrying as {candidate}")
                    continue
                else:
                    print(f"[ERROR] Failed to pull {candidate} (not found)")
                    import traceback; traceback.print_exc()
                    continue
            else:
                print(f"[ERROR] Failed to pull {candidate}")
                import traceback; traceback.print_exc()
                continue
        except Exception as e:
            last_error = e
            print(f"[ERROR] Failed to pull {candidate}")
            import traceback; traceback.print_exc()
            continue

        # success
        _ensure_model_defaults(candidate)
        if candidate not in models_state["models"]:
            models_state["models"].append(candidate)
            save_models()
        return cache_dir

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to pull {model_id}")

def _load(model_id: str, quantize: Optional[str] = None):
    """Load a model; on success add to models.json."""
    global pipe, current_model
    _unload()
    print(f"[Loading] Loading model: {model_id}")

    last_error = None
    resolved_id = None
    for candidate in _candidate_model_ids(model_id):
        _ensure_model_defaults(candidate)
        defaults = _merged_defaults_for(candidate)
        torch_dtype = _choose_dtype(defaults.get("dtype", DEFAULTS.get("dtype", "bfloat16")))
        pipeline_cls = _select_pipeline_cls(candidate)
        quant_config = _quantization_for(candidate, quantize=quantize)
        extra_kwargs = {}
        if quant_config is not None:
            extra_kwargs["quantization_config"] = quant_config
            extra_kwargs["device_map"] = "auto"

        def _load_pipeline(token: Optional[str]):
            return pipeline_cls.from_pretrained(
                candidate,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                token=token,
                **extra_kwargs,
            )

        try:
            pipe = _load_pipeline(HfFolder.get_token() or _env_hf_token())
            _apply_offload(pipe)
            resolved_id = candidate
            break
        except HfHubHTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code in (401, 403):
                print(f"[Auth] Access to {candidate} requires authentication.")
                token = _ensure_hf_auth(candidate)
                if not token:
                    continue
                try:
                    pipe = _load_pipeline(token)
                    _apply_offload(pipe)
                    resolved_id = candidate
                    break
                except Exception:
                    continue
            elif e.response is not None and e.response.status_code == 404:
                if candidate != model_id:
                    print(f"[Lookup] {model_id} not found; retrying as {candidate}")
                    continue
                else:
                    print(f"[ERROR] Failed to load {candidate} (not found)")
                    import traceback; traceback.print_exc()
                    continue
            else:
                print(f"[ERROR] Failed to load {candidate}")
                import traceback; traceback.print_exc()
                continue
        except Exception as e:
            last_error = e
            print(f"[ERROR] Failed to load {candidate}")
            import traceback; traceback.print_exc()
            continue

    if resolved_id is None:
        if last_error:
            raise last_error
        raise RuntimeError(f"Failed to load {model_id}")

    current_model = resolved_id
    models_state["loaded"] = resolved_id
    if resolved_id not in models_state["models"]:
        models_state["models"].append(resolved_id)
    save_models()
    return pipe



class GenRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    quantize: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None
    num_images: Optional[int] = None
    seed: Optional[int] = None

    # Server saving is OPT-IN via outfile; default None means: don't save on server
    outfile: Optional[str] = None

    # Encoding the server uses if it returns bytes (and to suggest extensions)
    format: Optional[str] = "png"       # "png" | "jpg" | "webp"

    # Should the server return images as base64 so the client can save locally?
    want_bytes: Optional[bool] = True


class ModelDefaultsRequest(BaseModel):
    model: str
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None
    num_images: Optional[int] = None
    seed: Optional[int] = None
    dtype: Optional[str] = None
    quantize: Optional[str] = None

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
        "model_defaults": MODEL_DEFAULTS,
    }


@app.get("/defaults/model")
def get_model_defaults(model: str = Query(..., description="Model id to inspect defaults")):
    return {
        "model": model,
        "defaults": _merged_defaults_for(model),
        "stored": _ensure_model_defaults(model),
    }


@app.post("/defaults/model")
def set_model_defaults(req: ModelDefaultsRequest):
    payload = req.dict(exclude_none=True)
    model_id = payload.pop("model")
    md = _update_model_defaults(model_id, payload)
    return {"status": "updated", "model": model_id, "defaults": md}

@app.get("/models")
def list_models():
    # Only the curated list
    return {"available": models_state["models"], "loaded": models_state["loaded"]}

@app.post("/load")
def load_model(
    model: str = Query(..., description="Repo id, e.g. black-forest-labs/FLUX.1-schnell"),
    quantize: Optional[str] = Query(None, description="Quantization mode: auto|none|bnb4|bnb8"),
):
    global pipe
    try:
        pipe = _load(model, quantize=quantize)
        return {
            "status": "loaded",
            "model": current_model,
            "quantize": quantize or (_ensure_model_defaults(model).get("quantize") or "auto"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {model}: {e}")


@app.post("/pull")
def pull_model(model: str = Query(..., description="Repo id to download without loading")):
    try:
        cache_dir = _pull_model(model)
        return {"status": "pulled", "model": model, "cache_dir": cache_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pull {model}: {e}")

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

def _clamp_and_seed(req, defaults: dict):
    steps = int(req.steps or defaults.get("steps", 4))
    width = int(req.width or defaults.get("width", 512))
    height = int(req.height or defaults.get("height", 512))
    guidance = float(req.guidance or defaults.get("guidance", 3.5))
    num_images = int(req.num_images or defaults.get("num_images", 1))
    seed = req.seed if req.seed is not None else defaults.get("seed", None)

    width = max(64, min(width, 1024))
    height = max(64, min(height, 1024))
    gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    return steps, width, height, guidance, num_images, gen

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
    model_id = _resolve_model(req.model)
    model_defaults = _merged_defaults_for(model_id)

    # Load (and add to list) if needed
    if current_model != model_id or pipe is None:
        try:
            pipe = _load(model_id, quantize=req.quantize)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load {model_id}: {e}")

    steps, width, height, guidance, num_images, generator = _clamp_and_seed(req, model_defaults)
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

    _reset_unload_timer()
    return {
        "status": "ok",
        "model": current_model,
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

