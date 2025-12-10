#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import base64
import math
import shutil
import subprocess
import tempfile
from PIL import Image
import base64 as b64mod

DEFAULT_SERVER = os.environ.get("FLUX_SERVER", "http://127.0.0.1:8000")

# ---------------- Preview helpers ---------------- #

def make_grid(images, cols, rows, padding=2, bg=(0,0,0,0)):
    img_w, img_h = images[0].size
    grid_w = cols * img_w + (cols - 1) * padding
    grid_h = rows * img_h + (rows - 1) * padding
    grid = Image.new("RGBA", (grid_w, grid_h), bg)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * (img_w + padding)
        y = r * (img_h + padding)
        grid.paste(img.convert("RGBA"), (x, y))
    return grid

def show_in_terminal(path):
    # Kitty
    if shutil.which("kitty"):
        try:
            subprocess.run(["kitty", "+kitten", "icat", path], check=True)
            return True
        except Exception:
            pass
    # iTerm2
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        with open(path, "rb") as f:
            data = b64mod.b64encode(f.read()).decode("ascii")
        sys.stdout.write(
            f"\033]1337;File=inline=1;width=auto;height=auto;preserveAspectRatio=1:{data}\a\n"
        )
        sys.stdout.flush()
        return True
    # Sixel
    if shutil.which("img2sixel"):
        subprocess.run(["img2sixel", path])
        return True
    return False

def preview_grid(paths, scale=0.5, padding=2):
    if not paths:
        return
    # load & scale
    images = []
    for f in paths:
        img = Image.open(f)
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(img)

    n = len(images)
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = make_grid(images, cols, rows, padding)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        grid.save(tmp_path, format="PNG")

    if not show_in_terminal(tmp_path):
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", tmp_path])
        elif shutil.which("open"):
            subprocess.Popen(["open", tmp_path])
        else:
            print(f"[Preview] Saved at {tmp_path} (open manually)")

# ---------------- Client UI ---------------- #

def pretty_kv(title, kv):
    print(title)
    for k, v in kv.items():
        print(f"  {k:16} {v}")
    print()

def pretty_list(title, items, bullet="•"):
    print(title)
    if not items:
        print("  (none)")
    else:
        for it in items:
            print(f"  {bullet} {it}")
    print()

def main():
    p = argparse.ArgumentParser(prog="flux_client", description="Client for flux_server")
    p.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="List tracked models and show which is loaded")
    sub.add_parser("defaults", help="Show server defaults")
    sub.add_parser("unload", help="Unload current model")

    loadp = sub.add_parser("load", help="Preload a model (adds to list if successful)")
    loadp.add_argument("model", help="Repo id, e.g. black-forest-labs/FLUX.1-schnell")
    loadp.add_argument("--quantize", choices=["none", "bnb4", "bnb8"], help="Quantization mode")
    loadp.add_argument("--quantize-text", choices=["none", "bnb4", "bnb8"], help="Quantization just for the text encoder")
    loadp.add_argument("--quantize-transformer", choices=["none", "bnb4", "bnb8"], help="Quantization just for the transformer/UNet")

    pullp = sub.add_parser("pull", help="Download a model without loading it")
    pullp.add_argument("model", help="Repo id to download")

    rmp = sub.add_parser("rm", help="Remove a model from list and delete its cache")
    rmp.add_argument("model", help="Repo id to remove")

    runp = sub.add_parser("run", help="Generate an image")
    runp.add_argument("prompt", nargs="+", help="Prompt text")
    runp.add_argument("-m", "--model", help="Repo id (auto-added on success)")
    runp.add_argument("-s", "--steps", type=int, help="Number of inference steps")
    runp.add_argument("-W", "--width", type=int, help="Image width")
    runp.add_argument("-H", "--height", type=int, help="Image height")
    runp.add_argument("-g", "--guidance", type=float, help="Guidance scale")
    runp.add_argument("-n", "--num-images", type=int, help="Number of images")
    runp.add_argument("-S", "--seed", type=int, help="Seed")
    runp.add_argument("-o", "--outfile", default="out.png", help="Output file (or prefix if -n>1)")
    runp.add_argument("-O", "--outdir", default=".", help="Directory to save images (client side)")
    runp.add_argument("-p", "--preview", action="store_true", help="Preview images in terminal")
    runp.add_argument("-q", "--quantize", choices=["none", "bnb4", "bnb8"], help="Quantization override")
    runp.add_argument("--quantize-text", choices=["none", "bnb4", "bnb8"], help="Quantization just for the text encoder")
    runp.add_argument("--quantize-transformer", choices=["none", "bnb4", "bnb8"], help="Quantization just for the transformer/UNet")

    args = p.parse_args()
    S = args.server.rstrip("/")

    try:
        if args.cmd == "ls":
            j = requests.get(f"{S}/models").json()
            loaded = j.get("loaded")
            avail = j.get("available", [])
            pretty_list("Available models:", avail)
            print(f"Loaded: {loaded if loaded else 'none'}")

        elif args.cmd == "defaults":
            j = requests.get(f"{S}/defaults").json()
            pretty_kv("Defaults:", j.get("defaults", {}))
            pretty_kv("Server:", {
                "unload_timeout": j.get("unload_timeout"),
                "memory_fraction": j.get("memory_fraction"),
                "default_model": j.get("default_model"),
                "server": S
            })

        elif args.cmd == "unload":
            requests.post(f"{S}/unload").json()
            print("Unloaded current model.")

        elif args.cmd == "load":
            params = {"model": args.model}
            if args.quantize:
                params["quantize"] = args.quantize
            if args.quantize_text:
                params["quantize_text"] = args.quantize_text
            if args.quantize_transformer:
                params["quantize_transformer"] = args.quantize_transformer
            j = requests.post(f"{S}/load", params=params).json()
            print(f"Loaded: {j.get('model')}")

        elif args.cmd == "pull":
            j = requests.post(f"{S}/pull", params={"model": args.model}).json()
            print(f"Pulled: {j.get('model')} -> {j.get('cache_dir')}")

        elif args.cmd == "rm":
            j = requests.delete(f"{S}/models", params={"model": args.model}).json()
            print(f"Removed: {j.get('model')}")

        elif args.cmd == "run":
            payload = {
                "prompt": " ".join(args.prompt),
                "want_bytes": True,
            }
            if args.model is not None:      payload["model"] = args.model
            if args.quantize is not None:   payload["quantize"] = None if args.quantize == "none" else args.quantize
            if args.quantize_text is not None: payload["quantize_text"] = None if args.quantize_text == "none" else args.quantize_text
            if args.quantize_transformer is not None: payload["quantize_transformer"] = None if args.quantize_transformer == "none" else args.quantize_transformer
            if args.steps is not None:      payload["steps"] = args.steps
            if args.width is not None:      payload["width"] = args.width
            if args.height is not None:     payload["height"] = args.height
            if args.guidance is not None:   payload["guidance"] = args.guidance
            if args.num_images is not None: payload["num_images"] = args.num_images
            if args.seed is not None:       payload["seed"] = args.seed
            if args.outfile != "out.png":   payload["outfile"] = args.outfile

            j = requests.post(f"{S}/generate", json=payload).json()
            files = j.get("files", [])
            images_b64 = j.get("images_b64", [])
            saved_server = j.get("saved", [])
            model = j.get("model")

            outdir = args.outdir
            os.makedirs(outdir, exist_ok=True)

            local_paths = []
            if images_b64 and files:
                for name, b64 in zip(files, images_b64):
                    path = os.path.join(outdir, name)
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    local_paths.append(path)
            elif saved_server:
                local_paths = saved_server

            print(f"Model: {model}")
            if local_paths:
                if len(local_paths) == 1:
                    print(f"Saved: {local_paths[0]}")
                else:
                    print("Saved:")
                    for pth in local_paths:
                        print(f"  • {pth}")

            if args.preview and local_paths:
                preview_grid(local_paths, scale=0.5)

        else:
            p.print_help()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

