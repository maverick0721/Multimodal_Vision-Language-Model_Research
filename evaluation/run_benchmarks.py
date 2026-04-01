import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from multimodal.vlm_model import VLM
from dataset.instruction_dataset import InstructionDataset
from utils.config import load_config

from evaluation.evaluate import (
    evaluate_caption,
    evaluate_vqa,
    evaluate_retrieval
)

print("Running benchmarks...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_cfg = load_config("configs/model.yaml")
vocab_size = int(model_cfg["vocab_size"])

# load model
model = VLM(vocab=vocab_size).to(device)
model.eval()
torch.set_grad_enabled(False)

ckpts = glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt")

if ckpts:
    latest = sorted(ckpts)[-1]
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("Loaded checkpoint:", latest)
else:
    print("No checkpoint found, using random weights.")

# dataset
dataset = InstructionDataset("data/instruction_data.json")

def collate_fn(batch):

    images = []
    tokens = []
    labels = []

    for item in batch:

        # dataset returns tuple
        if isinstance(item, tuple):

            if len(item) == 2:
                img, tok = item
                images.append(img)
                tokens.append(tok)

            elif len(item) == 3:
                img, tok, ans = item
                images.append(img)
                tokens.append(tok)
                labels.append(ans)

        # dataset returns dict
        elif isinstance(item, dict):

            images.append(item["image"])
            tokens.append(item["tokens"])

            if "labels" in item:
                labels.append(item["labels"])
            elif "answer" in item:
                labels.append(item["answer"])

    images = torch.stack(images)

    tokens = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=0
    )

    batch_out = {
        "image": images,
        "tokens": tokens
    }

    if len(labels) > 0:
        batch_out["labels"] = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

    return batch_out

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

# run benchmarks

print("Running caption benchmark...")
caption_score = evaluate_caption(model, loader, device)

print("Running VQA benchmark...")
vqa_score = evaluate_vqa(model, loader, device)

print("Running retrieval benchmark...")
retrieval_score = evaluate_retrieval(model, loader, device)

print("Caption BLEU:", caption_score)
print("VQA Accuracy:", vqa_score)
print("Retrieval Score:", retrieval_score)