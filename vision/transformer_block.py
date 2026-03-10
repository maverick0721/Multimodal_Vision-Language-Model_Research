import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import VisionAttention


class TransformerBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = VisionAttention(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        # attention
        x = x + checkpoint.checkpoint(
            lambda y: self.attn(self.norm1(y)),
            x
        )

        # feedforward
        x = x + checkpoint.checkpoint(
            lambda y: self.ffn(self.norm2(y)),
            x
        )

        return x