import torch
import torchvision.transforms as T
from PIL import Image

from multimodal.vlm_model import VLM
from inference.sampling import top_p


class VLMChat:

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


    def preprocess_image(self, path):

        img = Image.open(path).convert("RGB")

        img = self.transform(img)

        return img.unsqueeze(0).to(self.device)


    def tokenize(self, text, max_len=128):

        tokens = [ord(c) % 32000 for c in text]

        tokens = tokens[:max_len]

        return torch.tensor(tokens).unsqueeze(0).to(self.device)


    def detokenize(self, tokens):

        chars = [chr(int(t) % 256) for t in tokens]

        return "".join(chars)


    @torch.no_grad()
    def generate(self, image_path, prompt, max_tokens=64):

        image = self.preprocess_image(image_path)

        tokens = self.tokenize(prompt)

        for _ in range(max_tokens):

            logits = self.model(image, tokens)

            next_token = top_p(
                logits[:, -1, :],
                p=0.9,
                temp=0.8
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

        return self.detokenize(tokens[0])


if __name__ == "__main__":

    chat = VLMChat()

    while True:

        img = input("Image path: ")

        prompt = input("User: ")

        answer = chat.generate(img, prompt)

        print("Assistant:", answer)