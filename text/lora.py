import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, linear, r=8, alpha=16, dropout=0.0):

        super().__init__()

        assert isinstance(linear, nn.Linear)

        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = linear.in_features
        out_dim = linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

        # freeze base weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    # expose weight and bias so PyTorch modules still work
    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):

        base = self.linear(x)

        A = self.lora_A.to(x.device)
        B = self.lora_B.to(x.device)

        lora = self.dropout(x) @ A.t()
        lora = lora @ B.t()

        return base + lora * self.scaling


def apply_lora(module, target=("q", "k", "v", "out"), r=8, alpha=16):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear) and any(t in name for t in target):

            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))

        else:

            apply_lora(child, target, r, alpha)