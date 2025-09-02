#!/usr/bin/env bash
# Simple shell wrapper around the server & client
# Requires: python, jq (for pretty output), curl (if you want raw)
# Tip: export FLUX_SERVER=http://127.0.0.1:8000 to point to another host/port

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BIN="${HERE}/flux_server.py"
CLIENT_BIN="${HERE}/flux_client.py"
SERVER_URL="${FLUX_SERVER:-http://127.0.0.1:8000}"

usage() {
  cat <<'EOF'
flux - tiny model manager

Usage:
  flux serve [--host HOST] [--port PORT] [--config CONFIG]
  flux ls
  flux defaults
  flux unload
  flux load <model-id>
  flux rm <model-id>
  flux run "prompt text ..." [--model MODEL] [--steps N] [--width W] [--height H] [--guidance G] [--num-images K] [--seed SEED] [--outfile PATH] [--preview]
  flux -h

Notes:
  - Resolution: use --width and --height (e.g. 512x512 first).
  - Steps: --steps (e.g. 4–12 for quick previews).
  - Count: --num-images to get multiple outputs at once.
  - Guidance: --guidance (lower = freer, higher = stricter).
  - Seed: set for repeatability.
  - Preview: with Kitty terminal, add --preview to see inline.

Examples:
  flux serve --host 127.0.0.1 --port 8000
  flux ls
  flux run "A neon samurai in the rain" --steps 6 --width 512 --height 512 --preview
  flux load stabilityai/stable-diffusion-2-1
  flux rm black-forest-labs/FLUX.1-schnell
EOF
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  serve)
    host="127.0.0.1"
    port="8000"
    config="${HERE}/config.json"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --host) host="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
      esac
    done
    exec python "$SERVER_BIN" --host "$host" --port "$port" --config "$config"
    ;;

  ls)
    python "$CLIENT_BIN" --server "$SERVER_URL" ls
    ;;

  defaults)
    python "$CLIENT_BIN" --server "$SERVER_URL" defaults
    ;;

  unload)
    python "$CLIENT_BIN" --server "$SERVER_URL" unload
    ;;

  load)
    model="${1:-}"; [[ -z "$model" ]] && { echo "Missing <model-id>"; exit 1; }
    python "$CLIENT_BIN" --server "$SERVER_URL" load "$model"
    ;;

  rm)
    model="${1:-}"; [[ -z "$model" ]] && { echo "Missing <model-id>"; exit 1; }
    python "$CLIENT_BIN" --server "$SERVER_URL" rm "$model"
    ;;

  run)
    python "$CLIENT_BIN" --server "$SERVER_URL" run "$@"
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac

