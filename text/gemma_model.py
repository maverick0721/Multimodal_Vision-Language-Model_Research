import torch.nn as nn
from .decoder_layer import DecoderLayer

class GemmaModel(nn.Module):

    def __init__(self,vocab,dim=768,depth=12):

        super().__init__()

        self.embed = nn.Embedding(vocab,dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(dim) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim,vocab,bias=False)

        self.head.weight = self.embed.weight

    def forward(self, tokens, vision):

        x = self.embed(tokens)

        for layer in self.layers:

            x = layer(x, vision)

        x = self.norm(x)

        return self.head(x)