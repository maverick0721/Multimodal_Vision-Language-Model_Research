import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with LoRA adapters.
    During fine-tuning, freeze the base weight and train only A,B.
    """

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

        # Freeze base weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):

        base = self.linear(x)

        lora = self.dropout(x) @ self.lora_A.t()
        lora = lora @ self.lora_B.t()

        return base + lora * self.scaling


def apply_lora(module, target=("q", "k", "v", "out"), r=8, alpha=16):
    """
    Recursively replaces Linear layers in attention modules with LoRA versions.
    """

    for name, child in module.named_children():

        if isinstance(child, nn.Linear) and any(t in name for t in target):

            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))

        else:

            apply_lora(child, target, r=r, alpha=alpha)