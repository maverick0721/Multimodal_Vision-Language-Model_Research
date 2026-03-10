import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):

    def __init__(self, dim, hidden_dim):

        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):

        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):

    def __init__(self, dim, hidden_dim=3072, num_experts=4, top_k=2):

        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):

        B, T, D = x.shape

        logits = self.router(x)

        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(
            probs, self.top_k, dim=-1
        )

        out = torch.zeros_like(x)

        # token counts per expert
        expert_counts = torch.zeros(
            self.num_experts,
            device=x.device
        )

        for k in range(self.top_k):

            idx = topk_idx[..., k]
            prob = topk_probs[..., k].unsqueeze(-1)

            for expert_id in range(self.num_experts):

                mask = (idx == expert_id)

                if mask.any():

                    expert_counts[expert_id] += mask.sum()

                    expert_input = x[mask]

                    expert_out = self.experts[expert_id](expert_input)

                    out[mask] += prob[mask] * expert_out

        # compute load balancing loss
        router_prob_mean = probs.mean(dim=(0,1))

        token_fraction = expert_counts / expert_counts.sum()

        load_balance_loss = (
            self.num_experts *
            torch.sum(router_prob_mean * token_fraction)
        )

        return out, load_balance_loss