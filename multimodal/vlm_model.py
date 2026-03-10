import torch
import torch.nn as nn

from vision.siglip_encoder import SigLipEncoder
from vision.token_compressor import TokenCompressor
from text.gemma_model import GemmaModel
from multimodal.projector import ImageProjector
from multimodal.projection_heads import ProjectionHead


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

    def forward(self, image, tokens, kv_cache=None):

        # ----- vision encoder -----
        vision_tokens = self.vision(image)
        vision_tokens = self.compress(vision_tokens)
        vision_tokens = self.project(vision_tokens)

        # ----- text embeddings -----
        text_emb = self.text.embed(tokens)

        x = text_emb

        total_moe_loss = 0

        # ----- decoder layers -----
        for layer in self.text.layers:

            x, moe_loss = layer(x, vision_tokens)

            total_moe_loss += moe_loss

        x = self.text.norm(x)

        logits = self.text.head(x)

        # ----- embeddings for contrastive loss -----
        image_embed = vision_tokens.mean(dim=1)
        text_embed = text_emb.mean(dim=1)

        image_embed = self.image_proj(image_embed)
        text_embed = self.text_proj(text_embed)

        return logits, image_embed, text_embed, total_moe_loss