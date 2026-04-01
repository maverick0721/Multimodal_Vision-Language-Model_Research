import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from multimodal.vlm_model import VLM
from inference.paged_kv_cache import PagedKVCache
from inference.sampling import top_p
from utils.config import load_config
from dataset.instruction_format import build_prompt


class Generator:

    def __init__(self, checkpoint=None, vocab=None, device="cuda"):

        if vocab is None:
            model_cfg = load_config("configs/model.yaml")
            vocab = int(model_cfg["vocab_size"])

        self.device = device

        self.vocab = vocab
        self.model = VLM(vocab=vocab).to(device)

        if checkpoint is not None:

            ckpt = torch.load(checkpoint, map_location=device)

            self.model.load_state_dict(ckpt["model"])

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

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

        prompt = build_prompt(text)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        return encoded["input_ids"].to(self.device)


    def detokenize(self, tokens):

        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)


    def _apply_repetition_constraints(
        self,
        logits,
        generated_ids,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        max_repeat_token=2
    ):
        adjusted = logits.clone()

        if generated_ids.numel() > 0 and repetition_penalty > 1.0:
            unique_ids = generated_ids.unique()
            token_logits = adjusted[unique_ids]
            penalized = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty
            )
            adjusted[unique_ids] = penalized

        if generated_ids.numel() >= max_repeat_token:
            last_token = generated_ids[-1].item()
            recent = generated_ids[-max_repeat_token:]
            if torch.all(recent == last_token):
                adjusted[last_token] = -1e9

        if no_repeat_ngram_size and no_repeat_ngram_size > 1:
            n = no_repeat_ngram_size
            if generated_ids.numel() >= n - 1:
                ids = generated_ids.tolist()
                prefix = tuple(ids[-(n - 1):])
                banned = set()
                for i in range(len(ids) - n + 1):
                    if tuple(ids[i:i + n - 1]) == prefix:
                        banned.add(ids[i + n - 1])
                if banned:
                    adjusted[list(banned)] = -1e9

        if torch.isfinite(adjusted).any():
            return adjusted
        return logits


    def _cleanup_text(self, text):
        words = text.split()
        if not words:
            return text

        cleaned = []
        repeat_count = 0
        prev = None
        for w in words:
            lw = w.lower()
            if lw == prev:
                repeat_count += 1
            else:
                repeat_count = 1
                prev = lw

            if repeat_count <= 2:
                cleaned.append(w)

        return " ".join(cleaned).strip()


    @torch.no_grad()
    def generate(
        self,
        image_path,
        prompt,
        max_tokens=64,
        min_new_tokens=8,
        temperature=0.75,
        top_p_val=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        max_repeat_token=2
    ):

        image = self.preprocess_image(image_path)

        tokens = self.tokenize(prompt)
        prompt_len = tokens.shape[1]

        self.cache.reset()

        for _ in range(max_tokens):

            outputs = self.model(
                image=image,
                tokens=tokens,
                kv_cache=self.cache
            )

            # unpack model output
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]
            generated_ids = tokens[0][prompt_len:]
            constrained_logits = self._apply_repetition_constraints(
                next_logits[0],
                generated_ids,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_repeat_token=max_repeat_token
            )

            next_token = top_p(
                constrained_logits,
                p=top_p_val,
                temp=temperature
            ).unsqueeze(0)

            # Avoid terminating immediately with EOS so we can decode a non-empty answer.
            generated_so_far = tokens.shape[1] - prompt_len
            if generated_so_far < min_new_tokens and next_token.item() == self.eos_token_id:
                adjusted = constrained_logits.clone()
                adjusted[self.eos_token_id] = -1e9
                next_token = top_p(
                    adjusted,
                    p=top_p_val,
                    temp=temperature
                ).unsqueeze(0)

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

            if next_token.item() == self.eos_token_id and (tokens.shape[1] - prompt_len) >= min_new_tokens:
                break

        # return only the generated portion, excluding the prompt
        generated = tokens[0][prompt_len:]
        text = self._cleanup_text(self.detokenize(generated).strip())

        if text:
            return text

        # Fallback: re-run with a more deterministic decode profile if sampling returns blank text.
        self.cache.reset()
        tokens = self.tokenize(prompt)

        for _ in range(max_tokens):
            outputs = self.model(
                image=image,
                tokens=tokens,
                kv_cache=self.cache
            )

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]
            generated_ids = tokens[0][prompt_len:]
            constrained_logits = self._apply_repetition_constraints(
                next_logits[0],
                generated_ids,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_repeat_token=max_repeat_token
            )

            next_token = torch.argmax(constrained_logits.unsqueeze(0), dim=-1, keepdim=True)

            generated_so_far = tokens.shape[1] - prompt_len
            if generated_so_far < min_new_tokens and next_token.item() == self.eos_token_id:
                adjusted = constrained_logits.unsqueeze(0).clone()
                adjusted[:, self.eos_token_id] = -1e9
                next_token = torch.argmax(adjusted, dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == self.eos_token_id and (tokens.shape[1] - prompt_len) >= min_new_tokens:
                break

        generated = tokens[0][prompt_len:]
        text = self._cleanup_text(self.detokenize(generated).strip())
        return text if text else "No answer generated. Try another image or prompt."


if __name__ == "__main__":

    import sys
    import os
    import glob
    import readline  # enables backspace, arrow keys, and line editing in input()

    # Fix terminal so backspace/arrow keys work in VS Code and other terminals
    if sys.stdin.isatty():
        os.system("stty sane")

    ckpts = sorted(glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt"))
    checkpoint = ckpts[-1] if ckpts else None
    if checkpoint:
        print("Using checkpoint:", checkpoint)

    gen = Generator(checkpoint=checkpoint)

    while True:
        try:
            img = input("image path: ").strip()
            prompt = input("prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not img or not prompt:
            print("Please provide both image path and prompt.")
            continue
        out = gen.generate(img, prompt)
        print("\nModel:", out)
        try:
            cont = input("\nContinue? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cont != "y":
            break