#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-demo}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20}"
WEB_PORT="${WEB_PORT:-7860}"

load_env_file() {
    if [ -f ".env" ]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi

    if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
        export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    fi

    if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
        export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
    fi
}

print_usage() {
    cat <<'EOF'
Usage: ./scripts/start_project.sh [demo|full|explain]

Modes:
  demo     Run a fast end-to-end pipeline check, then start web demo (default)
  full     Run the full pipeline, then start web demo
  explain  Print project walkthrough only (no execution)

Environment overrides:
  TRAIN_MAX_STEPS=20   Max training steps used in demo mode
  WEB_PORT=7860        Port for web demo
  WEB_DEMO_BACKEND=blip|vlm   Backend for web_demo.py
EOF
}

project_walkthrough() {
    cat <<EOF

=================================================
Multimodal Vision-Language Model: Quick Walkthrough
=================================================
1) Data + images are read from data/instruction_data.json and data/images/
2) Model training runs via training/train_vlm.py
3) Evaluation + benchmarks run under evaluation/
4) Web demo is launched from web_demo.py
5) Open in browser: http://127.0.0.1:${WEB_PORT}

For a full run use:
  ./scripts/start_project.sh full

For fast demo run (default):
  ./scripts/start_project.sh demo

EOF
}

ensure_venv() {
    if [ -d ".venv" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "ERROR: .venv not found in ${ROOT_DIR}"
        exit 1
    fi
}

stop_existing_web_demo() {
    pkill -f "python web_demo.py" >/dev/null 2>&1 || true
}

start_web_demo() {
    stop_existing_web_demo
    echo "Starting web demo on http://127.0.0.1:${WEB_PORT}"
    WEB_DEMO_PORT="$WEB_PORT" python web_demo.py
}

if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    print_usage
    exit 0
fi

if [ "$MODE" = "explain" ]; then
    project_walkthrough
    exit 0
fi

if [ "$MODE" != "demo" ] && [ "$MODE" != "full" ]; then
    echo "ERROR: Unknown mode '$MODE'"
    print_usage
    exit 1
fi

load_env_file
ensure_venv
project_walkthrough

if [ "$MODE" = "demo" ]; then
    echo "Running fast pipeline for demonstration..."
    FAST_DRY_RUN=1 TRAIN_MAX_STEPS="$TRAIN_MAX_STEPS" RUN_INTERACTIVE=0 ./scripts/run_all.sh
else
    echo "Running full pipeline..."
    RUN_INTERACTIVE=0 ./scripts/run_all.sh
fi

start_web_demo