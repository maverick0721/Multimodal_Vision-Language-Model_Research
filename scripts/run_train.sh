#!/bin/bash
echo "===== TRAINING VLM ====="

OMP_NUM_THREADS=1 \
torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  -m training.train_vlm \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True
