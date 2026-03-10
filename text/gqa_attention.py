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

    def forward(self, x, kv_cache=None, layer_id=None):

        B, T, D = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        if kv_cache is not None:

            prev_k, prev_v = kv_cache.get(layer_id)

            if prev_k is not None:

                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)

            kv_cache.append(layer_id, k, v)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1,2).reshape(B, T, D)

        return self.out(out)