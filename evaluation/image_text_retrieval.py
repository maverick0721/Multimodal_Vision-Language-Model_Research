import torch

def recall_at_k(sim_matrix,k):

    correct = 0

    for i in range(sim_matrix.size(0)):

        topk = torch.topk(
            sim_matrix[i],
            k
        ).indices

        if i in topk:
            correct += 1

    return correct / sim_matrix.size(0)