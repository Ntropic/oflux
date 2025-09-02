#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import base64

from shutil import which

DEFAULT_SERVER = os.environ.get("FLUX_SERVER", "http://127.0.0.1:8000")

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
    runp.add_argument("-p", "--preview", action="store_true", help="Preview via Kitty icat if available")

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
            j = requests.post(f"{S}/unload").json()
            print("Unloaded current model.")

        elif args.cmd == "load":
            j = requests.post(f"{S}/load", params={"model": args.model}).json()
            print(f"Loaded: {j.get('model')}")

        elif args.cmd == "rm":
            j = requests.delete(f"{S}/models", params={"model": args.model}).json()
            print(f"Removed: {j.get('model')}")

        elif args.cmd == "run":
            payload = {
                "prompt": " ".join(args.prompt),
                "want_bytes": True,   # ask server to return image bytes
                # don't send 'outfile' unless user explicitly set it
            }
            if args.model is not None:      payload["model"] = args.model
            if args.steps is not None:      payload["steps"] = args.steps
            if args.width is not None:      payload["width"] = args.width
            if args.height is not None:     payload["height"] = args.height
            if args.guidance is not None:   payload["guidance"] = args.guidance
            if args.num_images is not None: payload["num_images"] = args.num_images
            if args.seed is not None:       payload["seed"] = args.seed
            # only send outfile if user changed it from default behavior
            if args.outfile != "out.png":
                payload["outfile"] = args.outfile

            j = requests.post(f"{S}/generate", json=payload).json()
            files = j.get("files", [])             # suggested filenames (no path)
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
                # If server saved (explicit outfile) but didn't send bytes, just show server paths
                local_paths = saved_server

            print(f"Model: {model}")
            if local_paths:
                if len(local_paths) == 1:
                    print(f"Saved: {local_paths[0]}")
                else:
                    print("Saved:")
                    for pth in local_paths:
                        print(f"  • {pth}")

            if args.preview and os.environ.get("TERM", "").startswith("xterm-kitty") and which("kitty"):
                if local_paths:
                    os.system("kitty +kitten icat " + " ".join(local_paths))
        else:
            p.print_help()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

