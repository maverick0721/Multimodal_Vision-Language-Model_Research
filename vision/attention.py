import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, dim, heads=8):

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)

        self.out = nn.Linear(dim, dim)

    def forward(self, x):

        B, T, D = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = qkv

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1,2).reshape(B, T, D)

        return self.out(out)