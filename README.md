# Oflux

**Oflux** is a lightweight server + CLI client to run diffusion models (FLUX, Stable Diffusion) locally.  
Inspired by [ollama](https://ollama.com/), it instead provides a simple interface for image generation.  


## Install

```bash
git clone https://github.com/Ntropic/oflux.git
cd oflux
chmod +x oflux
./oflux  # first run guides dependency install and PATH setup
```

On first launch, the wrapper auto-installs Python requirements (prompting for the ROCm variant when interactive) and offers to symlink `oflux` into `~/.local/bin` so new shells can find it. If you start the installer from an active conda environment, that environment name is recorded and re-activated on future runs when possible.

## Usage

Start the service:

```bash
oflux serve [--host HOST] [--port PORT] [--config PATH] [--models PATH]
```

List currently downloaded models:

```bash
oflux ls
```

Prefetch a gated or large model without loading it:

```bash
oflux pull black-forest-labs/FLUX.1-schnell
```

If a repo cannot be found, `oflux` automatically retries by prefixing `black-forest-labs/` to the requested model name.

Generate an image:

```bash
oflux run "a cinematic dragon flying over snowy mountains"
```
<p align="center">
  <img src="cinematic_dragon_flying_over_snowy_mountains_seed1844931357.png" alt="Dragon" width="512"/>
</p>

Optional options for `run`
| Argument        | Description |
|-----------------|-------------|
| `-m, --model MODEL`   | Model repo id to use (auto-added if downloaded) |
| `-s, --steps N`       | Inference steps (e.g. 4–12 for previews) |
| `-W, --width PX`      | Image width (default 512) |
| `-H, --height PX`     | Image height (default 512) |
| `-g, --guidance G`    | Guidance scale (default 3.5) |
| `-n, --num-images K`  | Number of images (default 1) |
| `-S, --seed N`        | Seed for reproducibility |
| `-o, --outfile PATH`  | Output file or prefix (if `-n>1`). If missing will auto-generate a name from prompt |
| `-O, --outdir DIR`    | Directory to save images (client side, default `.`) |
| `-p, --preview`       | Preview inline (Kitty terminal only) |
| `-q, --quantize MODE` | Per-call quantization override (`auto`, `none`, `bnb4`, `bnb8`) |

### Model defaults

* Each model keeps its own default settings (steps, size, guidance, dtype, quantization, etc.). These are stored under `model_defaults` in `config.json` so switching models automatically restores the saved defaults.
* When no model is specified and nothing is currently loaded, the server now falls back to the configured `default_model` (ignoring any stale `loaded` value from previous sessions).
* Defaults are created the first time you pull or load a model. FLUX.2 models start with a `bnb4` quantization default to fit easier on GPUs.
* Update a model’s defaults via CLI:

```bash
oflux set-defaults black-forest-labs/FLUX.2-dev --quantize none --steps 6 --width 1024 --height 1024
```
Use `--quantize none` to disable quantization for that model or `--quantize auto` to clear an override.

### Quantization

* Quantization defaults live per model (see above). Per-call overrides via `-q/--quantize` take precedence for that invocation only.
* Allowed values: `bnb4`, `bnb8`, `none`, or `auto` (auto = use saved defaults). Use `none` to load the model without quantization even if a default exists.
* Requires the `bitsandbytes` dependency (installed automatically on first run).


#### Other commands:

```bash
oflux defaults
oflux set-defaults <model-id>
oflux unload
oflux load <model-id>
oflux pull <model-id>
oflux rm <model-id>
oflux -h
```

## Notes

* Models are downloaded from HuggingFace automatically.
* Idle models are unloaded after inactivity (`config.json`, default **300s**).
* Currently tested on **AMD GPUs with ROCm**, Nvidia should work as well.

## Author
Michael Schilling