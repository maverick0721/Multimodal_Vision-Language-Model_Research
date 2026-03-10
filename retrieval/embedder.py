import torch
import torch.nn as nn


class SimpleEmbedder(nn.Module):

    def __init__(self, vocab=32000, dim=768):

        super().__init__()

        self.emb = nn.Embedding(vocab, dim)

    def forward(self, texts):

        ids = torch.randint(0,32000,(len(texts),16))

        return self.emb(ids).mean(dim=1)