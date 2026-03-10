import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, linear, r=8, alpha=16):

        super().__init__()

        self.linear = linear

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(
            torch.randn(r, in_features) * 0.01
        )

        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r)
        )

        self.scale = alpha / r

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        base = self.linear(x)

        lora = self.dropout(x) @ self.lora_A.t()

        lora = lora @ self.lora_B.t()

        return base + self.scale * lora


def apply_lora(module, target=("Linear",), r=8, alpha=16):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):

            setattr(
                module,
                name,
                LoRALinear(child, r=r, alpha=alpha)
            )

        else:

            apply_lora(child, target, r, alpha)