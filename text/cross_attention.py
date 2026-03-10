import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, dim, heads=8):

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x, vision):

        B, T, D = x.shape
        V = vision.size(1)

        q = self.q(x)
        k = self.k(vision)
        v = self.v(vision)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, V, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, V, self.heads, self.head_dim).transpose(1,2)

        attn = F.scaled_dot_product_attention(q, k, v)

        out = attn.transpose(1,2).reshape(B, T, D)

        return self.out(out)