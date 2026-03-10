import torch.nn as nn

from vision.siglip_encoder import SigLipEncoder
from vision.token_compressor import TokenCompressor
from text.gemma_model import GemmaModel
from .projector import ImageProjector


class VLM(nn.Module):

    def __init__(self,vocab):

        super().__init__()

        self.vision = SigLipEncoder()

        self.compress = TokenCompressor()

        self.project = ImageProjector(768,768)

        self.text = GemmaModel(vocab)

    def forward(self,image,tokens):

        vision_tokens = self.vision(image)

        vision_tokens = self.compress(vision_tokens)

        vision_tokens = self.project(vision_tokens)

        logits = self.text(tokens)

        return logits