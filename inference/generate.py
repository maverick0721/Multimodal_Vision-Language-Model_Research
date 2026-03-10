# inference/generate.py

import torch
from PIL import Image
import torchvision.transforms as T

from multimodal.vlm_model import VLM
from inference.paged_kv_cache import PagedKVCache
from inference.sampling import top_p


class Generator:

    def __init__(self, checkpoint=None, vocab=32000, device="cuda"):

        self.device = device

        self.model = VLM(vocab=vocab).to(device)

        if checkpoint is not None:

            ckpt = torch.load(checkpoint, map_location=device)

            self.model.load_state_dict(ckpt["model"])

        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])

        self.cache = PagedKVCache(
            layers=len(self.model.text.layers),
            heads=8,
            head_dim=96
        )


    def preprocess_image(self, path):

        img = Image.open(path).convert("RGB")

        img = self.transform(img)

        return img.unsqueeze(0).to(self.device)


    def tokenize(self, text):

        tokens = [ord(c) % 32000 for c in text]

        return torch.tensor(tokens).unsqueeze(0).to(self.device)


    def detokenize(self, tokens):

        chars = [chr(int(t) % 256) for t in tokens]

        return "".join(chars)


    @torch.no_grad()
    def generate(
        self,
        image_path,
        prompt,
        max_tokens=64,
        temperature=0.8,
        top_p_val=0.9
    ):

        image = self.preprocess_image(image_path)

        tokens = self.tokenize(prompt)

        self.cache.reset()

        for _ in range(max_tokens):

            logits = self.model(
                image,
                tokens,
                kv_cache=self.cache
            )

            next_token = top_p(
                logits[:, -1, :],
                p=top_p_val,
                temp=temperature
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

        return self.detokenize(tokens[0])


if __name__ == "__main__":

    gen = Generator()

    while True:

        img = input("image path: ")

        prompt = input("prompt: ")

        out = gen.generate(img, prompt)

        print("\nModel:", out)