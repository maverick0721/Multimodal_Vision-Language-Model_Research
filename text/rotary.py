import torch

def rotary(q,k):

    dim = q.shape[-1]

    freqs = torch.arange(dim//2,device=q.device)

    theta = 10000 ** (-2*freqs/dim)

    sin = torch.sin(theta)
    cos = torch.cos(theta)

    q1,q2 = q[...,::2],q[...,1::2]

    q = torch.cat(
        [q1*cos-q2*sin,
         q1*sin+q2*cos],
        dim=-1
    )

    return q,k