#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
from shutil import which

DEFAULT_SERVER = os.environ.get("FLUX_SERVER", "http://127.0.0.1:8000")

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser(prog="flux_client", description="Client for flux_server")
    p.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="List models and show which is loaded")
    sub.add_parser("defaults", help="Show server defaults")
    sub.add_parser("unload", help="Unload current model")

    loadp = sub.add_parser("load", help="Preload a model")
    loadp.add_argument("model", help="Repo id")

    rmp = sub.add_parser("rm", help="Remove a model from cache")
    rmp.add_argument("model", help="Repo id to remove")

    runp = sub.add_parser("run", help="Generate an image")
    runp.add_argument("prompt", nargs="+", help="Prompt text")
    runp.add_argument("--model", help="Repo id")
    runp.add_argument("--steps", type=int)
    runp.add_argument("--width", type=int)
    runp.add_argument("--height", type=int)
    runp.add_argument("--guidance", type=float)
    runp.add_argument("--num-images", type=int)
    runp.add_argument("--seed", type=int)
    runp.add_argument("--outfile", default="out.png")
    runp.add_argument("--preview", action="store_true", help="Preview via Kitty icat if available")

    args = p.parse_args()
    S = args.server.rstrip("/")

    try:
        if args.cmd == "ls":
            resp = requests.get(f"{S}/models")
            resp.raise_for_status()
            pretty(resp.json())

        elif args.cmd == "defaults":
            resp = requests.get(f"{S}/defaults")
            resp.raise_for_status()
            pretty(resp.json())

        elif args.cmd == "unload":
            resp = requests.post(f"{S}/unload")
            resp.raise_for_status()
            pretty(resp.json())

        elif args.cmd == "load":
            resp = requests.post(f"{S}/load", params={"model": args.model})
            resp.raise_for_status()
            pretty(resp.json())

        elif args.cmd == "rm":
            resp = requests.delete(f"{S}/models", params={"model": args.model})
            resp.raise_for_status()
            pretty(resp.json())

        elif args.cmd == "run":
            payload = {
                "prompt": " ".join(args.prompt),
                "outfile": args.outfile
            }
            for k in ("model", "steps", "width", "height", "guidance", "num_images", "seed"):
                v = getattr(args, k.replace("-", "_"))
                if v is not None:
                    payload[k.replace("-", "_")] = v

            resp = requests.post(f"{S}/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            pretty(data)

            if args.preview and os.environ.get("TERM", "").startswith("xterm-kitty") and which("kitty"):
                # show all saved images
                files = data.get("saved", [])
                if files:
                    os.system("kitty +kitten icat " + " ".join(files))

        else:
            p.print_help()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

