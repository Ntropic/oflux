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

List currently downloaded models:

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
```
  -m, --model MODEL        Model repo id to use (auto-added if downloaded)
  -s, --steps N            Inference steps (e.g. 4–12 for previews)
  -W, --width PX           Image width (default 512)
  -H, --height PX          Image height (default 512)
  -g, --guidance G         Guidance scale (default 3.5)
  -n, --num-images K       Number of images (default 1)
  -S, --seed N             Seed for reproducibility
  -o, --outfile PATH       Output file or prefix (if -n>1). If missing will auto generate a name from prompt
  -O, --outdir DIR         Directory to save images (client side, default .)
  -p, --preview            Preview inline (Kitty terminal only)
```

#### Other commands:

```bash
oflux defaults
oflux unload
oflux load <model-id>
oflux rm <model-id>
oflux -h
```

## Notes

* Models are downloaded from HuggingFace automatically.
* Idle models are unloaded after inactivity (`config.json`, default **300s**).
* Currently tested on **AMD GPUs with ROCm**, Nvidia should work as well.

## Author
Michael Schilling