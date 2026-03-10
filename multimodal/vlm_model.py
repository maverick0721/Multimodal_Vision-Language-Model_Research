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

        vision_tokens = self.vision(image)

        vision_tokens = self.compress(vision_tokens)

        vision_tokens = self.project(vision_tokens)

        text_emb = self.text.embed(tokens)

        x = text_emb

        for layer in self.text.layers:

            x = layer(x, vision_tokens)

        x = self.text.norm(x)

        logits = self.text.head(x)

        return logits