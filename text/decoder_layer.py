import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .rmsnorm import RMSNorm
from .gqa_attention import GQAAttention
from .cross_attention import CrossAttention
from .moe_ffn import MoEFFN


class DecoderLayer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.attn = GQAAttention(dim)

        self.cross = CrossAttention(dim)

        self.ffn = MoEFFN(dim)

    def forward(self, x, vision_tokens):

        # self attention
        x = x + checkpoint.checkpoint(
            lambda y: self.attn(self.norm1(y)),
            x
        )

        # cross attention with vision tokens
        x = x + checkpoint.checkpoint(
            lambda y: self.cross(self.norm2(y), vision_tokens),
            x
        )

        # mixture of experts feedforward
        x = x + checkpoint.checkpoint(
            lambda y: self.ffn(self.norm3(y)),
            x
        )

        return x