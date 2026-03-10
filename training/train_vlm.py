import os
import glob
import json
import torch
import argparse

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from multimodal.vlm_model import VLM
from utils.config import load_config
from experiments.logger import Logger


# -----------------------------
# CUDA stability
# -----------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# Checkpoint loader
# -----------------------------

def load_latest_checkpoint(model, optimizer, output_dir):

    ckpts = glob.glob(os.path.join(output_dir, "checkpoint_*.pt"))

    if len(ckpts) == 0:
        return 0

    ckpts.sort()

    latest = ckpts[-1]

    print("Resuming from:", latest)

    data = torch.load(latest)

    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])

    return data["step"]


# -----------------------------
# Arguments
# -----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--model_config")
parser.add_argument("--train_config")
parser.add_argument("--output_dir")

args = parser.parse_args()


# -----------------------------
# Load configs
# -----------------------------

model_cfg = load_config(args.model_config)
train_cfg = load_config(args.train_config)


# -----------------------------
# Training parameters
# -----------------------------

batch_size = train_cfg["batch_size"]
epochs = train_cfg["epochs"]

accum_steps = train_cfg.get("gradient_accumulation_steps", 1)


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model
# -----------------------------

model = VLM(
    vision_dim=model_cfg["vision_dim"],
    text_dim=model_cfg["text_dim"],
    num_layers=model_cfg["num_layers"],
    vocab_size=model_cfg["vocab_size"]
)

model = model.to(device)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

scaler = GradScaler()


# -----------------------------
# Logger
# -----------------------------

os.makedirs(args.output_dir, exist_ok=True)

logger = Logger(args.output_dir)


# -----------------------------
# Dataset placeholder
# -----------------------------

dataset = []

loader = DataLoader(
    dataset,
    batch_size=batch_size
)


# -----------------------------
# Resume checkpoint
# -----------------------------

start_step = load_latest_checkpoint(
    model,
    optimizer,
    args.output_dir
)


# -----------------------------
# Training loop
# -----------------------------

step = start_step

optimizer.zero_grad()

for epoch in range(epochs):

    for batch in loader:

        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)

        # Mixed precision forward
        with autocast():

            logits = model(images, tokens)

            loss = logits.mean()

        # Normalize loss for accumulation
        loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Update weights after accumulation
        if step % accum_steps == 0:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        logger.log(step, loss.item())

        # Logging
        if step % 10 == 0:

            vram = 0

            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e9

            print(
                f"step {step} | loss {loss.item():.4f} | "
                f"vram {vram:.2f} GB"
            )

        # Save checkpoint
        if step % 1000 == 0:

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step
                },
                os.path.join(
                    args.output_dir,
                    f"checkpoint_{step}.pt"
                )
            )

        step += 1


# -----------------------------
# Save metrics
# -----------------------------

metrics = {
    "final_loss": float(loss.item())
}

with open(
    os.path.join(args.output_dir, "metrics.json"),
    "w"
) as f:

    json.dump(metrics, f)