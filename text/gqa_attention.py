import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):

    def __init__(self, dim, heads=8):

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x):

        B, T, D = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        attn = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        out = out.transpose(1,2).reshape(B, T, D)

        return self.out(out)