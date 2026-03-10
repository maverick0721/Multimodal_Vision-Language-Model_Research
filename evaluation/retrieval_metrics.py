import torch

def recall_at_k(similarity, k=5):

    ranks = similarity.argsort(dim=1, descending=True)

    correct = torch.arange(similarity.size(0)).unsqueeze(1)

    hits = (ranks[:, :k] == correct).any(dim=1)

    return hits.float().mean().item()