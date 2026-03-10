#!/bin/bash

echo "===== Multimodal Vision-Language Pipeline ====="

echo ""
echo "Activating environment..."

source .venv/bin/activate

echo ""
echo "Checking GPU..."

python - << EOF
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "Select pipeline mode:"
echo "1) Train model"
echo "2) Run inference"
echo "3) Start chat interface"
echo "4) Train + Chat"
echo ""

read -p "Enter option: " MODE

if [ "$MODE" == "1" ]; then
    ./scripts/run_train.sh

elif [ "$MODE" == "2" ]; then
    ./scripts/run_inference.sh

elif [ "$MODE" == "3" ]; then
    ./scripts/run_chat.sh

elif [ "$MODE" == "4" ]; then
    ./scripts/run_train.sh
    ./scripts/run_chat.sh

else
    echo "Invalid option"
fi