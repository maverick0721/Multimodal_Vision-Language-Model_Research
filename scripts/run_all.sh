#!/bin/bash

set -e

echo "================================================="
echo "   Multimodal Vision Language Model Pipeline"
echo "================================================="


echo ""
echo "STEP 1: Activate environment"
echo "-------------------------------------------------"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found"
    exit 1
fi


echo ""
echo "STEP 2: GPU check"
echo "-------------------------------------------------"

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF


echo ""
echo "STEP 3: Dataset check"
echo "-------------------------------------------------"

if [ ! -f "data/instruction_data.json" ]; then
    echo "ERROR: dataset missing -> data/instruction_data.json"
    exit 1
fi

if [ ! -d "data/images" ]; then
    echo "ERROR: image folder missing -> data/images"
    exit 1
fi

echo "Dataset OK"


echo ""
echo "STEP 4: Training model"
echo "-------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --output_dir outputs


echo ""
echo "STEP 5: Evaluation"
echo "-------------------------------------------------"

python -m evaluation.evaluate


echo ""
echo "STEP 6: Benchmarks"
echo "-------------------------------------------------"

python -m evaluation.run_benchmarks


echo ""
echo "STEP 7: Demo test"
echo "-------------------------------------------------"

if [ -f "demo.py" ]; then
    python -m demo
else
    echo "demo.py not found, skipping demo"
fi


echo ""
echo "STEP 8: Smoke test"
echo "-------------------------------------------------"

if [ -f "scripts/smoke_test.py" ]; then
    python scripts/smoke_test.py
else
    echo "scripts/smoke_test.py not found, skipping smoke test"
fi


echo ""
echo "STEP 9: Inference test"
echo "-------------------------------------------------"

if [ "${RUN_INTERACTIVE:-0}" = "1" ] && [ -f "inference/generate.py" ]; then
    python -m inference.generate
else
    echo "Skipping interactive inference (set RUN_INTERACTIVE=1 to enable)"
fi


echo ""
echo "STEP 10: Chat interface"
echo "-------------------------------------------------"

if [ "${RUN_INTERACTIVE:-0}" = "1" ] && [ -f "inference/run_chat.py" ]; then
    python -m inference.run_chat
else
    echo "Skipping interactive chat (set RUN_INTERACTIVE=1 to enable)"
fi


echo ""
echo "================================================="
echo "PIPELINE FINISHED SUCCESSFULLY"
echo "================================================="