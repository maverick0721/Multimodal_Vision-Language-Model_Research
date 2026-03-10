#!/bin/bash

EXP=experiments/run_$(date +%s)

mkdir -p $EXP

echo "Running experiment in $EXP"

python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --output_dir $EXP