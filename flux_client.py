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
from typing import Any, Dict, Optional

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


def _clean_last_payload(last: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not last:
        return {}
    cleaned = {k: v for k, v in last.items() if k not in {"timestamp"}}
    return cleaned


def perform_generation(server_url: str, payload: Dict[str, Any], outdir: str, preview: bool) -> None:
    j = requests.post(f"{server_url}/generate", json=payload).json()
    files = j.get("files", [])
    images_b64 = j.get("images_b64", [])
    saved_server = j.get("saved", [])
    model = j.get("model")
    lora = j.get("lora")
    lora_scale = j.get("lora_scale")

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
    if lora:
        scale_str = f" (scale {lora_scale})" if lora_scale is not None else ""
        print(f"LoRA: {lora}{scale_str}")
    else:
        print("LoRA: none")
    if local_paths:
        if len(local_paths) == 1:
            print(f"Saved: {local_paths[0]}")
        else:
            print("Saved:")
            for pth in local_paths:
                print(f"  • {pth}")

    if preview and local_paths:
        preview_grid(local_paths, scale=0.5)


def run_interactive_cli(server_url: str) -> None:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.current import get_app
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, VSplit
        from prompt_toolkit.shortcuts import message_dialog
        from prompt_toolkit.widgets import Button, Checkbox, Dialog, Frame, Label, RadioList, TextArea
    except ImportError as exc:
        print("prompt_toolkit is required for the interactive CLI. Install it via 'pip install prompt_toolkit'.", file=sys.stderr)
        sys.exit(1)

    models_info = requests.get(f"{server_url}/models").json()
    defaults_info = requests.get(f"{server_url}/defaults").json()
    last_info = requests.get(f"{server_url}/last").json()

    defaults = defaults_info.get("defaults", {})
    default_model = defaults_info.get("default_model")
    default_lora = defaults_info.get("default_lora")

    available_models = models_info.get("available", [])
    loaded_model = models_info.get("loaded")
    available_loras = models_info.get("loras", [])
    active_lora = models_info.get("active_lora")
    active_lora_scale = models_info.get("active_lora_scale")

    last_settings = last_info.get("last") if last_info else None

    model_options = []
    seen_models = set()

    def add_model_option(value: Optional[str], label: str) -> None:
        if value in seen_models:
            return
        seen_models.add(value)
        model_options.append((value, label))

    if default_model:
        note = " (default)"
        if default_model not in available_models:
            note = " (default • downloads on first use)"
        add_model_option(default_model, f"⭐ {default_model}{note}")

    for mid in available_models:
        label = mid
        if mid == loaded_model:
            label += " [loaded]"
        add_model_option(mid, label)

    if not model_options:
        model_options.append((None, "No cached models yet"))

    initial_model = (last_settings or {}).get("model") or loaded_model or default_model or model_options[0][0]

    model_radio = RadioList(model_options)
    try:
        model_radio.current_value = initial_model
    except Exception:
        if model_options:
            model_radio.current_value = model_options[0][0]

    custom_model_input = TextArea(height=1, multiline=False)

    lora_options = []
    seen_loras = set()

    def add_lora_option(value: Optional[str], label: str) -> None:
        key = value or "__none__"
        if key in seen_loras:
            return
        seen_loras.add(key)
        lora_options.append((value, label))

    add_lora_option(None, "No LoRA (use base pipeline)")

    if default_lora:
        note = " (default)"
        if default_lora not in available_loras:
            note = " (default • downloads on first use)"
        add_lora_option(default_lora, f"⭐ {default_lora}{note}")

    for lid in available_loras:
        label = lid
        if lid == active_lora:
            label += " [active]"
        add_lora_option(lid, label)

    initial_lora = (last_settings or {}).get("lora")
    if initial_lora is None:
        initial_lora = active_lora if active_lora is not None else default_lora

    lora_radio = RadioList(lora_options)
    try:
        lora_radio.current_value = initial_lora
    except Exception:
        if lora_options:
            lora_radio.current_value = lora_options[0][0]

    custom_lora_input = TextArea(height=1, multiline=False)

    prompt_text = (last_settings or {}).get("prompt") or ""
    prompt_area = TextArea(text=prompt_text, height=5, multiline=True)

    steps_default = str((last_settings or {}).get("steps") or defaults.get("steps", 4))
    width_default = str((last_settings or {}).get("width") or defaults.get("width", 512))
    height_default = str((last_settings or {}).get("height") or defaults.get("height", 512))
    guidance_default = str((last_settings or {}).get("guidance") or defaults.get("guidance", 3.5))
    num_images_default = str((last_settings or {}).get("num_images") or defaults.get("num_images", 1))
    seed_val = (last_settings or {}).get("seed")
    seed_default = "" if seed_val is None else str(seed_val)
    lora_scale_default = str((last_settings or {}).get("lora_scale") or active_lora_scale or 1.0)

    steps_field = TextArea(text=steps_default, height=1, multiline=False)
    width_field = TextArea(text=width_default, height=1, multiline=False)
    height_field = TextArea(text=height_default, height=1, multiline=False)
    guidance_field = TextArea(text=guidance_default, height=1, multiline=False)
    num_images_field = TextArea(text=num_images_default, height=1, multiline=False)
    seed_field = TextArea(text=seed_default, height=1, multiline=False)
    lora_scale_field = TextArea(text=lora_scale_default, height=1, multiline=False)
    outfile_field = TextArea(text=(last_settings or {}).get("outfile") or "", height=1, multiline=False)
    outdir_field = TextArea(text=".", height=1, multiline=False)
    preview_checkbox = Checkbox("Preview images after generation", checked=False)

    def apply_last(_: Optional[Button]) -> None:
        if not last_settings:
            message_dialog(title="No history", text="Run at least once before using the last settings option.").run()
            return
        prompt_area.text = last_settings.get("prompt") or ""
        last_model = last_settings.get("model")
        if last_model in seen_models:
            model_radio.current_value = last_model
            custom_model_input.text = ""
        elif last_model:
            custom_model_input.text = last_model

        last_lora = last_settings.get("lora")
        key = last_lora if last_lora is not None else None
        if (last_lora is None and None in [opt[0] for opt in lora_options]) or key in [opt[0] for opt in lora_options]:
            lora_radio.current_value = key
            custom_lora_input.text = ""
        elif last_lora:
            custom_lora_input.text = last_lora

        steps_field.text = str(last_settings.get("steps") or defaults.get("steps", 4))
        width_field.text = str(last_settings.get("width") or defaults.get("width", 512))
        height_field.text = str(last_settings.get("height") or defaults.get("height", 512))
        guidance_field.text = str(last_settings.get("guidance") or defaults.get("guidance", 3.5))
        num_images_field.text = str(last_settings.get("num_images") or defaults.get("num_images", 1))
        seed_val = last_settings.get("seed")
        seed_field.text = "" if seed_val is None else str(seed_val)
        lora_scale_field.text = str(last_settings.get("lora_scale") or 1.0)
        outfile_field.text = last_settings.get("outfile") or ""

    def reset_defaults(_: Optional[Button]) -> None:
        prompt_area.text = ""
        if default_model in seen_models:
            model_radio.current_value = default_model
        elif model_options:
            model_radio.current_value = model_options[0][0]
        custom_model_input.text = ""

        if default_lora in [opt[0] for opt in lora_options]:
            lora_radio.current_value = default_lora
        else:
            lora_radio.current_value = None
        custom_lora_input.text = ""

        steps_field.text = str(defaults.get("steps", 4))
        width_field.text = str(defaults.get("width", 512))
        height_field.text = str(defaults.get("height", 512))
        guidance_field.text = str(defaults.get("guidance", 3.5))
        num_images_field.text = str(defaults.get("num_images", 1))
        seed_field.text = ""
        lora_scale_field.text = "1.0"
        outfile_field.text = ""
        outdir_field.text = "."
        preview_checkbox.checked = False

    def on_cancel(_: Optional[Button]) -> None:
        get_app().exit(result=None)

    def on_generate(_: Optional[Button]) -> None:
        prompt_value = prompt_area.text.strip()
        if not prompt_value:
            message_dialog(title="Prompt required", text="Please enter a prompt before generating.").run()
            return

        model_value = model_radio.current_value
        custom_model = custom_model_input.text.strip()
        if custom_model:
            model_value = custom_model

        if not model_value:
            message_dialog(title="Model required", text="Select or enter a model repo id to continue.").run()
            return

        lora_value = lora_radio.current_value
        custom_lora = custom_lora_input.text.strip()
        if custom_lora:
            lora_value = custom_lora

        def parse_int(field: TextArea, label: str, minimum: int = 1, default: Optional[int] = None) -> Optional[int]:
            raw = field.text.strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                message_dialog(title="Invalid value", text=f"{label} must be an integer.").run()
                return None
            if value < minimum:
                message_dialog(title="Invalid value", text=f"{label} must be >= {minimum}.").run()
                return None
            return value

        def parse_float(field: TextArea, label: str, default: Optional[float] = None) -> Optional[float]:
            raw = field.text.strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                message_dialog(title="Invalid value", text=f"{label} must be a number.").run()
                return None

        steps_value = parse_int(steps_field, "Steps", minimum=1, default=defaults.get("steps", 4))
        if steps_value is None:
            return
        width_value = parse_int(width_field, "Width", minimum=64, default=defaults.get("width", 512))
        if width_value is None:
            return
        height_value = parse_int(height_field, "Height", minimum=64, default=defaults.get("height", 512))
        if height_value is None:
            return
        guidance_value = parse_float(guidance_field, "Guidance", default=float(defaults.get("guidance", 3.5)))
        if guidance_value is None:
            return
        num_images_value = parse_int(num_images_field, "Number of images", minimum=1, default=defaults.get("num_images", 1))
        if num_images_value is None:
            return

        seed_text = seed_field.text.strip()
        if seed_text:
            try:
                seed_value = int(seed_text)
            except ValueError:
                message_dialog(title="Invalid value", text="Seed must be an integer.").run()
                return
        else:
            seed_value = None

        lora_scale_text = lora_scale_field.text.strip()
        if lora_scale_text:
            try:
                lora_scale_value = float(lora_scale_text)
            except ValueError:
                message_dialog(title="Invalid value", text="LoRA scale must be a number.").run()
                return
        else:
            lora_scale_value = None

        outfile_text = outfile_field.text.strip()
        outdir_text = outdir_field.text.strip() or "."

        payload: Dict[str, Any] = {
            "prompt": prompt_value,
            "model": model_value,
            "steps": steps_value,
            "width": width_value,
            "height": height_value,
            "guidance": guidance_value,
            "num_images": num_images_value,
            "want_bytes": True,
        }

        if seed_value is not None:
            payload["seed"] = seed_value
        payload["lora"] = lora_value
        if lora_scale_value is not None:
            payload["lora_scale"] = lora_scale_value
        if outfile_text:
            payload["outfile"] = outfile_text

        result = {
            "payload": payload,
            "outdir": outdir_text,
            "preview": preview_checkbox.checked,
        }
        get_app().exit(result=result)

    model_frame = Frame(title="Models", body=HSplit([
        model_radio,
        Label("Enter a repo id to override the selection:"),
        custom_model_input,
    ], padding=1))

    lora_frame = Frame(title="LoRA adapters", body=HSplit([
        lora_radio,
        Label("Enter a repo id to override the selection:"),
        custom_lora_input,
    ], padding=1))

    tuning_frame = Frame(title="Generation settings", body=HSplit([
        VSplit([
            HSplit([Label("Steps"), steps_field]),
            HSplit([Label("Width"), width_field]),
            HSplit([Label("Height"), height_field]),
        ], padding=2),
        VSplit([
            HSplit([Label("Guidance"), guidance_field]),
            HSplit([Label("Images"), num_images_field]),
            HSplit([Label("Seed"), seed_field]),
        ], padding=2),
        VSplit([
            HSplit([Label("LoRA scale"), lora_scale_field]),
            HSplit([Label("Outfile prefix"), outfile_field]),
            HSplit([Label("Output directory"), outdir_field]),
        ], padding=2),
        preview_checkbox,
    ], padding=1))

    body = HSplit([
        Label("Enter a prompt (Shift+Enter for newline):"),
        prompt_area,
        VSplit([model_frame, lora_frame], padding=2),
        tuning_frame,
    ], padding=1)

    buttons = [
        Button(text="Use last", handler=apply_last),
        Button(text="Reset", handler=reset_defaults),
        Button(text="Generate", handler=on_generate),
        Button(text="Cancel", handler=on_cancel),
    ]

    dialog = Dialog(title="oflux interactive CLI", body=body, buttons=buttons, with_background=True)

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):  # type: ignore
        event.app.exit(result=None)

    app = Application(layout=Layout(dialog), key_bindings=kb, full_screen=True)
    result = app.run()

    if not result:
        print("Interactive session cancelled.")
        return

    payload = result["payload"]
    outdir = result["outdir"]
    preview = result["preview"]

    perform_generation(server_url, payload, outdir, preview)


def main():
    p = argparse.ArgumentParser(prog="flux_client", description="Client for flux_server")
    p.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="List tracked models and show which is loaded")
    sub.add_parser("defaults", help="Show server defaults")
    sub.add_parser("unload", help="Unload current model")
    sub.add_parser("cli", help="Launch an interactive terminal UI for generation")

    loadp = sub.add_parser("load", help="Preload a model (adds to list if successful)")
    loadp.add_argument("model", help="Repo id, e.g. black-forest-labs/FLUX.1-schnell")

    rmp = sub.add_parser("rm", help="Remove a model (or LoRA) from the local cache")
    rmp.add_argument("model", help="Repo id to remove")
    rmp.add_argument("--lora", action="store_true", help="Treat repo id as a LoRA adapter")

    runp = sub.add_parser("run", help="Generate an image")
    runp.add_argument("prompt", nargs="*", help="Prompt text (optional with --last)")
    runp.add_argument("-m", "--model", help="Repo id (auto-added on success)")
    runp.add_argument("-s", "--steps", type=int, help="Number of inference steps")
    runp.add_argument("-W", "--width", type=int, help="Image width")
    runp.add_argument("-H", "--height", type=int, help="Image height")
    runp.add_argument("-g", "--guidance", type=float, help="Guidance scale")
    runp.add_argument("-n", "--num-images", type=int, help="Number of images")
    runp.add_argument("-S", "--seed", type=int, help="Seed")
    runp.add_argument("-L", "--lora", help="LoRA adapter repo id to apply")
    runp.add_argument("--lora-scale", type=float, help="Scale to apply to the LoRA adapter (default 1.0)")
    runp.add_argument("-o", "--outfile", default="out.png", help="Output file (or prefix if -n>1)")
    runp.add_argument("-O", "--outdir", default=".", help="Directory to save images (client side)")
    runp.add_argument("-p", "--preview", action="store_true", help="Preview images in terminal")
    runp.add_argument("-l", "--last", action="store_true", help="Reuse the last generation settings saved on the server")

    args = p.parse_args()
    S = args.server.rstrip("/")

    try:
        if args.cmd == "ls":
            j = requests.get(f"{S}/models").json()
            loaded = j.get("loaded")
            avail = j.get("available", [])
            loras = j.get("loras", [])
            pretty_list("Available models:", avail)
            print(f"Loaded model: {loaded if loaded else 'none'}\n")
            pretty_list("Downloaded LoRA adapters:", loras)
            active_lora = j.get("active_lora")
            if active_lora:
                scale = j.get("active_lora_scale")
                scale_txt = f" (scale {scale})" if scale is not None else ""
                print(f"Active LoRA: {active_lora}{scale_txt}")
            else:
                print("Active LoRA: none")
            print()

        elif args.cmd == "defaults":
            j = requests.get(f"{S}/defaults").json()
            pretty_kv("Defaults:", j.get("defaults", {}))
            pretty_kv("Server:", {
                "unload_timeout": j.get("unload_timeout"),
                "memory_fraction": j.get("memory_fraction"),
                "default_model": j.get("default_model"),
                "default_lora": j.get("default_lora"),
                "server": S
            })

        elif args.cmd == "unload":
            requests.post(f"{S}/unload").json()
            print("Unloaded current model.")

        elif args.cmd == "load":
            j = requests.post(f"{S}/load", params={"model": args.model}).json()
            print(f"Loaded: {j.get('model')}")

        elif args.cmd == "rm":
            if args.lora:
                j = requests.delete(f"{S}/loras", params={"lora": args.model}).json()
                print(f"Removed LoRA: {j.get('lora')}")
            else:
                j = requests.delete(f"{S}/models", params={"model": args.model}).json()
                print(f"Removed model: {j.get('model')}")

        elif args.cmd == "cli":
            run_interactive_cli(S)

        elif args.cmd == "run":
            payload: Dict[str, Any] = {}

            if args.last:
                last_resp = requests.get(f"{S}/last").json()
                last_payload = _clean_last_payload(last_resp.get("last")) if last_resp else {}
                if not last_payload:
                    print("No previous generation recorded on the server. Run once without --last first.", file=sys.stderr)
                    sys.exit(1)
                payload.update(last_payload)

            prompt_override = " ".join(args.prompt) if args.prompt else None
            if prompt_override:
                payload["prompt"] = prompt_override

            if not payload.get("prompt"):
                print("Prompt text is required unless --last finds a stored prompt.", file=sys.stderr)
                sys.exit(1)

            if args.model is not None:
                payload["model"] = args.model
            if args.steps is not None:
                payload["steps"] = args.steps
            if args.width is not None:
                payload["width"] = args.width
            if args.height is not None:
                payload["height"] = args.height
            if args.guidance is not None:
                payload["guidance"] = args.guidance
            if args.num_images is not None:
                payload["num_images"] = args.num_images
            if args.seed is not None:
                payload["seed"] = args.seed
            if args.lora is not None:
                payload["lora"] = args.lora
            if args.lora_scale is not None:
                payload["lora_scale"] = args.lora_scale
            if args.outfile != "out.png":
                payload["outfile"] = args.outfile

            payload["want_bytes"] = True

            perform_generation(S, payload, args.outdir, args.preview)

        else:
            p.print_help()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

