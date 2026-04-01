import os
import glob
import json
import torch
import argparse

from torch.utils.data import DataLoader

from multimodal.vlm_model import VLM
from utils.config import load_config
from experiments.logger import Logger
from dataset.instruction_dataset import InstructionDataset, collate_fn


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

    for latest in reversed(ckpts):

        try:
            print("Resuming from:", latest)

            data = torch.load(latest, map_location="cpu")

            model.load_state_dict(data["model"])
            optimizer.load_state_dict(data["optimizer"])

            return data["step"]

        except Exception as e:
            print(f"Skipping unreadable checkpoint {latest}: {e}")

    print("No valid checkpoint found, starting from scratch.")
    return 0


# -----------------------------
# Arguments
# -----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--model_config", type=str, default="configs/model.yaml")
parser.add_argument("--train_config", type=str, default="configs/training.yaml")
parser.add_argument("--optim_config", type=str, default="configs/optimizer.yaml")
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--max_steps", type=int, default=-1)

args = parser.parse_args()


# -----------------------------
# Load configs
# -----------------------------

model_cfg = load_config(args.model_config)
train_cfg = load_config(args.train_config)
optim_cfg = load_config(args.optim_config)


# -----------------------------
# Training parameters
# -----------------------------

batch_size = train_cfg["batch_size"]
epochs = train_cfg["epochs"]

contrastive_weight = train_cfg.get("contrastive_weight", 0.1)
moe_weight = train_cfg.get("moe_weight", 0.01)

vocab_size = int(model_cfg["vocab_size"])


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model
# -----------------------------

model = VLM(
    vocab=vocab_size,
    dim=int(model_cfg["text_dim"])
).to(device)

print("Model vocab:", vocab_size)
print("Embedding shape:", model.text.embed.weight.shape)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(optim_cfg["lr"]),
    weight_decay=float(optim_cfg["weight_decay"]),
    betas=tuple(optim_cfg["betas"])
)


# -----------------------------
# Logger
# -----------------------------

os.makedirs(args.output_dir, exist_ok=True)
logger = Logger(args.output_dir)


# -----------------------------
# Dataset
# -----------------------------

dataset = InstructionDataset("data/instruction_data.json")

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

print("Dataset size:", len(dataset))
print("Total batches:", len(loader))


# -----------------------------
# Resume checkpoint
# -----------------------------

start_step = load_latest_checkpoint(
    model,
    optimizer,
    args.output_dir
)

step = start_step
max_steps = int(args.max_steps)
last_loss = None
stop_training = False

optimizer.zero_grad()


# -----------------------------
# Training loop
# -----------------------------

for epoch in range(epochs):

    for batch in loader:

        if max_steps >= 0 and step >= max_steps:
            stop_training = True
            break

        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device).long()

        # -----------------------------
        # Label safety
        # -----------------------------

        labels = torch.where(labels >= vocab_size, -100, labels)
        labels = torch.where(labels < -100, -100, labels)

        # -----------------------------
        # Forward pass
        # -----------------------------

        logits, img_emb, txt_emb, moe_loss = model(images, tokens)

        # Ensure shapes are valid
        if img_emb.dim() > 2:
            img_emb = img_emb.mean(dim=1)

        if txt_emb.dim() > 2:
            txt_emb = txt_emb.mean(dim=1)

        # -----------------------------
        # Language modeling loss (shifted for next-token prediction)
        # -----------------------------

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        lm_loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100
        )

        # -----------------------------
        # Contrastive loss
        # -----------------------------

        img_emb = torch.nan_to_num(img_emb.float())
        txt_emb = torch.nan_to_num(txt_emb.float())

        img_emb = torch.nn.functional.normalize(img_emb, dim=-1, eps=1e-6)
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1, eps=1e-6)

        similarity = img_emb @ txt_emb.T

        targets = torch.arange(similarity.size(0), device=device)

        contrastive_loss = torch.nn.functional.cross_entropy(
            similarity,
            targets
        )

        # -----------------------------
        # Total loss
        # -----------------------------

        loss = (
            lm_loss
            + contrastive_weight * contrastive_loss
            + moe_weight * moe_loss
        )

        last_loss = loss.item()

        # -----------------------------
        # Backprop
        # -----------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        logger.log(step, loss.item())

        # -----------------------------
        # Logging
        # -----------------------------

        if step % 10 == 0:

            vram = 0

            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e9

            print(
                f"step {step} | loss {loss.item():.4f} | "
                f"vram {vram:.2f} GB"
            )

        # -----------------------------
        # Checkpoint
        # -----------------------------

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

    if stop_training:
        print(f"Reached max_steps={max_steps}, stopping early.")
        break


# -----------------------------
# Save final checkpoint
# -----------------------------

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

print(f"Saved final checkpoint at step {step}")


# -----------------------------
# Save metrics
# -----------------------------

metrics = {
    "final_loss": float(last_loss) if last_loss else None
}

with open(
    os.path.join(args.output_dir, "metrics.json"),
    "w"
) as f:
    json.dump(metrics, f)

print("Training finished.")