import json
import torch
from PIL import Image
import torchvision.transforms as T

from dataset.instruction_format import build_prompt


transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])


def tokenize(text):

    tokens = [ord(c) % 32000 for c in text]

    return torch.tensor(tokens)


class InstructionDataset:

    def __init__(self, json_path):

        with open(json_path) as f:

            self.data = json.load(f)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        image = Image.open(sample["image"]).convert("RGB")

        image = transform(image)

        question = sample["conversations"][0]["content"]

        answer = sample["conversations"][1]["content"]

        prompt = build_prompt(question)

        prompt_tokens = tokenize(prompt)

        answer_tokens = tokenize(answer)

        tokens = torch.cat([prompt_tokens, answer_tokens])

        labels = torch.cat([
            torch.full_like(prompt_tokens, -100),
            answer_tokens
        ])

        return image, tokens, labels