import torch
import torch.nn.functional as F


class SimpleRetriever:

    def __init__(self, texts, embedder):

        self.texts = texts
        self.embedder = embedder

        self.embeddings = embedder(texts)

    def search(self, query, k=3):

        q = self.embedder([query])

        sims = F.cosine_similarity(q, self.embeddings)

        idx = torch.topk(sims, k).indices

        return [self.texts[i] for i in idx]