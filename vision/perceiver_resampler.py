import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):

    def __init__(self, dim=768, num_latents=64, heads=8):

        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(num_latents, dim)
        )

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.heads = heads
        self.head_dim = dim // heads

        self.out = nn.Linear(dim, dim)

    def forward(self, vision):

        B, T, D = vision.shape

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        q = self.q(latents)
        k = self.k(vision)
        v = self.v(vision)

        L = latents.size(1)

        q = q.view(B, L, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        attn = F.scaled_dot_product_attention(q, k, v)

        out = attn.transpose(1,2).reshape(B, L, D)

        return self.out(out)