import torch
import torch.nn as nn

from vision.siglip_encoder import SigLipEncoder
from vision.token_compressor import TokenCompressor
from vision.perceiver_resampler import PerceiverResampler
from text.gemma_model import GemmaModel
from multimodal.projector import ImageProjector
from multimodal.projection_heads import ProjectionHead

self.resampler = PerceiverResampler()

class VLM(nn.Module):

    def __init__(self, vocab, dim=768):

        super().__init__()

        # vision stack
        self.vision = SigLipEncoder()
        self.compress = TokenCompressor()
        self.project = ImageProjector(768, dim)

        # language model
        self.text = GemmaModel(vocab, dim)

        # projection heads for contrastive learning
        self.image_proj = ProjectionHead(dim)
        self.text_proj = ProjectionHead(dim)

    def forward(self, image, tokens):

        vision_tokens = self.vision(image)

        vision_tokens = self.resampler(vision_tokens)

        vision_tokens = self.project(vision_tokens)

        logits = self.text(tokens, vision_tokens)

        img_emb = vision_tokens.mean(dim=1)

        txt_emb = logits.mean(dim=1)

        moe_loss = torch.tensor(0.0, device=image.device)

        return logits, img_emb, txt_emb, moe_loss