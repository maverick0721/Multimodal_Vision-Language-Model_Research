#!/bin/bash

set -e

run_cmd() {
    if [ "${DRY_RUN_COMMANDS:-0}" = "1" ]; then
        echo "[DRY-RUN] $*"
        return 0
    fi
    "$@"
}

echo "================================================="
echo "   Multimodal Vision Language Model Pipeline"
echo "================================================="


echo ""
echo "STEP 1: Activate environment"
echo "-------------------------------------------------"

if [ "${SKIP_VENV_CHECK:-0}" = "1" ]; then
    echo "Skipping virtual environment activation"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found"
    exit 1
fi


echo ""
echo "STEP 2: GPU check"
echo "-------------------------------------------------"

if [ "${SKIP_GPU_CHECK:-0}" = "1" ]; then
    echo "Skipping GPU check"
else
run_cmd python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
fi


echo ""
echo "STEP 3: Dataset check"
echo "-------------------------------------------------"

if [ "${SKIP_DATASET_CHECK:-0}" = "1" ]; then
    echo "Skipping dataset check"
else
    if [ ! -f "data/instruction_data.json" ]; then
        echo "ERROR: dataset missing -> data/instruction_data.json"
        exit 1
    fi

    if [ ! -d "data/images" ]; then
        echo "ERROR: image folder missing -> data/images"
        exit 1
    fi
fi

echo "Dataset OK"


echo ""
echo "STEP 3.5: Checkpoint hygiene"
echo "-------------------------------------------------"

if [ -d "outputs" ]; then
run_cmd python - <<EOF
import glob
import os
import shutil
import torch

checkpoints = sorted(glob.glob("outputs/checkpoint_*.pt"))
if not checkpoints:
    print("No checkpoints found")
    raise SystemExit(0)

quarantine_dir = "outputs/quarantine"
os.makedirs(quarantine_dir, exist_ok=True)

quarantined = 0
for ckpt in checkpoints:
    try:
        torch.load(ckpt, map_location="cpu")
    except Exception as exc:
        dst = os.path.join(quarantine_dir, os.path.basename(ckpt))
        if os.path.exists(dst):
            dst = os.path.join(quarantine_dir, os.path.basename(ckpt) + ".bad")
        shutil.move(ckpt, dst)
        quarantined += 1
        print(f"Quarantined {ckpt}: {exc}")

if quarantined == 0:
    print("Checkpoint scan OK")
else:
    print(f"Quarantined {quarantined} checkpoint(s)")
EOF
else
    echo "No outputs directory found, skipping checkpoint hygiene"
fi


FAST_DRY_RUN="${FAST_DRY_RUN:-0}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20}"
TRAIN_STEPS_ARG="-1"

if [ "$FAST_DRY_RUN" = "1" ]; then
    TRAIN_STEPS_ARG="$TRAIN_MAX_STEPS"
    RUN_INTERACTIVE=0
    echo "FAST_DRY_RUN enabled: training capped at ${TRAIN_STEPS_ARG} steps"
fi


echo ""
echo "STEP 4: Training model"
echo "-------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:$(pwd)

run_cmd python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --output_dir outputs \
    --max_steps "$TRAIN_STEPS_ARG"


echo ""
echo "STEP 5: Evaluation"
echo "-------------------------------------------------"

run_cmd python -m evaluation.evaluate


echo ""
echo "STEP 6: Benchmarks"
echo "-------------------------------------------------"

run_cmd python -m evaluation.run_benchmarks


echo ""
echo "STEP 7: Demo test"
echo "-------------------------------------------------"

if [ -f "demo.py" ]; then
    run_cmd python -m demo
else
    echo "demo.py not found, skipping demo"
fi


echo ""
echo "STEP 8: Smoke test"
echo "-------------------------------------------------"

if [ -f "scripts/smoke_test.py" ]; then
    run_cmd python scripts/smoke_test.py
else
    echo "scripts/smoke_test.py not found, skipping smoke test"
fi


echo ""
echo "STEP 9: Inference test"
echo "-------------------------------------------------"

if [ "${RUN_INTERACTIVE:-0}" = "1" ] && [ -f "inference/generate.py" ]; then
    run_cmd python -m inference.generate
else
    echo "Skipping interactive inference (set RUN_INTERACTIVE=1 to enable)"
fi


echo ""
echo "STEP 10: Chat interface"
echo "-------------------------------------------------"

if [ "${RUN_INTERACTIVE:-0}" = "1" ] && [ -f "inference/run_chat.py" ]; then
    run_cmd python -m inference.run_chat
else
    echo "Skipping interactive chat (set RUN_INTERACTIVE=1 to enable)"
fi


echo ""
echo "================================================="
echo "PIPELINE FINISHED SUCCESSFULLY"
echo "================================================="