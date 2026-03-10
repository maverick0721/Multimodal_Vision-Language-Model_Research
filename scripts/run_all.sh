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

python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --output_dir experiments/run_pipeline


echo ""
echo "STEP 5: Evaluation"
echo "-------------------------------------------------"

python evaluation/evaluate.py


echo ""
echo "STEP 6: Benchmarks"
echo "-------------------------------------------------"

python evaluation/run_benchmarks.py


echo ""
echo "STEP 7: Demo test"
echo "-------------------------------------------------"

if [ -f "demo.py" ]; then
    python demo.py
else
    echo "demo.py not found, skipping demo"
fi


echo ""
echo "STEP 8: Inference test"
echo "-------------------------------------------------"

if [ -f "inference/generate.py" ]; then
    python inference/generate.py
fi


echo ""
echo "STEP 9: Chat interface"
echo "-------------------------------------------------"

if [ -f "inference/run_chat.py" ]; then
    python inference/run_chat.py
fi


echo ""
echo "================================================="
echo "PIPELINE FINISHED SUCCESSFULLY"
echo "================================================="