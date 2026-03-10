import torch.nn as nn

from .gqa_attention import GQAAttention
from .cross_attention import CrossAttention
from .moe_ffn import MoEFFN
from .rmsnorm import RMSNorm


class DecoderLayer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = GQAAttention(dim)

        self.norm2 = RMSNorm(dim)
        self.cross = CrossAttention(dim)

        self.norm3 = RMSNorm(dim)

        self.ffn = MoEFFN(
            dim,
            hidden_dim=3072,
            num_experts=4,
            top_k=2
        )


    def forward(self, x, vision):

        # self attention
        x = x + self.attn(self.norm1(x))

        # cross attention (vision)
        x = x + self.cross(self.norm2(x), vision)

        # mixture-of-experts FFN
        ffn_out, moe_loss = self.ffn(self.norm3(x))

        x = x + ffn_out

        return x, moe_loss