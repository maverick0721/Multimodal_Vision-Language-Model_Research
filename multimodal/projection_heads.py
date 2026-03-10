import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(self, dim, proj_dim=512):

        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):

        return self.proj(x)