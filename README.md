# Oflux

**Oflux** is a lightweight server + CLI client to run diffusion models (FLUX, Stable Diffusion) locally.  
Inspired by [ollama](https://ollama.com/), it instead provides a simple interface for image generation.  


## Install

```bash
git clone https://github.com/Ntropic/oflux.git
cd oflux
conda create -n oflux python=3.11 -y
conda activate oflux
pip install -r requirements.txt  # optionally: pip install -r requirements_rocm.txt
chmod +x oflux 
```

## Usage

Start the service:

```bash
oflux serve [--host HOST] [--port PORT] [--config PATH] [--models PATH]
```

List currently downloaded models and cached LoRA adapters:

```bash
oflux ls
```

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
| `-L, --lora ID`       | Hugging Face LoRA adapter to apply |
| `    --lora-scale S`  | Scale to use when applying the LoRA (default 1.0) |
| `-l, --last`          | Reuse the last settings saved on the server (prompt optional) |
| `-o, --outfile PATH`  | Output file or prefix (if `-n>1`). If missing will auto-generate a name from prompt |
| `-O, --outdir DIR`    | Directory to save images (client side, default `.`) |
| `-p, --preview`       | Preview inline (Kitty terminal only) |

Use `oflux cli` for an interactive terminal UI that lets you pick models, LoRA adapters, tweak generation settings, and enter a prompt with arrow keys or your mouse. The UI highlights the recommended defaults—even before they're downloaded—and you can load or reset to the last successful run with a single button press.

Every successful generation is persisted on the server (model, LoRA, steps, resolution, etc.), making it easy to repeat it later with `oflux run --last` or by pressing **Use last** inside the interactive CLI.


#### Other commands:

```bash
oflux defaults
oflux unload
oflux load <model-id>
oflux rm <model-id>
oflux rm --lora <adapter-id>
oflux cli
oflux -h
```

## Notes

* Models are downloaded from HuggingFace automatically.
* Idle models are unloaded after inactivity (`config.json`, default **300s**).
* Currently tested on **AMD GPUs with ROCm**, Nvidia should work as well.

## Author
Michael Schilling