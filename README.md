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
| `-q, --quantize MODE` | Quantize both text + transformer (`none`, `bnb4`, `bnb8`) |
| `--quantize-text MODE` | Quantize just the text encoder (`none`, `bnb4`, `bnb8`) |
| `--quantize-transformer MODE` | Quantize just the transformer/UNet (`none`, `bnb4`, `bnb8`) |

### Quantization

* The server chooses quantization based on `config.json`. By default, any model containing `flux.2` is loaded in 4-bit (bitsandbytes) to fit on more GPUs. Override per run with `-q`/`--quantize` or set `"quantization"` in `config.json` (e.g., `{ "flux.2": "bnb4" }`).
* You can quantize components separately. Use `--quantize-text` (text encoder) and/or `--quantize-transformer` (UNet/transformer). If only `-q/--quantize` is provided it applies to both components.
* Requires the `bitsandbytes` dependency (installed automatically on first run).


#### Other commands:

```bash
oflux defaults
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