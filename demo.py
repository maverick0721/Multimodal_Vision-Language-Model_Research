import torch
import glob

from inference.generate import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpts = sorted(glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt"))

if len(ckpts) == 0:
    raise RuntimeError("No checkpoints found")

checkpoint = None

for candidate in reversed(ckpts):
    try:
        torch.load(candidate, map_location="cpu")
        checkpoint = candidate
        break
    except Exception as e:
        print(f"Skipping unreadable checkpoint {candidate}: {e}")

if checkpoint is None:
    raise RuntimeError("No valid checkpoint found")

print("Using checkpoint:", checkpoint)

generator = Generator(checkpoint)

prompt = "What animal is this?"

image_path = "images/dog.jpg"

output = generator.generate(image_path, prompt)

print("Model output:", output)