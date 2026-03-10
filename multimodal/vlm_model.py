import torch
import torch.nn as nn

from vision.siglip_encoder import SigLipEncoder
from vision.token_compressor import TokenCompressor
from text.gemma_model import GemmaModel
from .projector import ImageProjector


class VLM(nn.Module):

    def __init__(self, vocab, dim=768):

        super().__init__()

        self.vision = SigLipEncoder()

        self.compress = TokenCompressor()

        self.project = ImageProjector(768, dim)

        self.text = GemmaModel(vocab, dim)

    def forward(self, image, tokens):

        # Vision encoder
        vision_tokens = self.vision(image)

        # Compress vision tokens
        vision_tokens = self.compress(vision_tokens)

        # Project into text embedding space
        vision_tokens = self.project(vision_tokens)

        # Text embeddings
        text_emb = self.text.embed(tokens)

        # Concatenate vision + text tokens
        x = torch.cat([vision_tokens, text_emb], dim=1)

        # Pass through decoder layers
        for layer in self.text.layers:
            x = layer(x)

        x = self.text.norm(x)

        logits = self.text.head(x)

        # Return only text token predictions
        vision_len = vision_tokens.shape[1]

        return logits[:, vision_len:, :]