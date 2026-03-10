import torch
import torch.nn as nn
from .ffn import FFN

class MoE(nn.Module):

    def __init__(self,dim,experts=4):

        super().__init__()

        self.router = nn.Linear(dim,experts)

        self.experts = nn.ModuleList(
            [FFN(dim) for _ in range(experts)]
        )

    def forward(self,x):

        w = torch.softmax(self.router(x),-1)

        out = 0

        for i,e in enumerate(self.experts):
            out += e(x)*w[...,i:i+1]

        return out