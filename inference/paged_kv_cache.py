import torch


class PagedKVCache:

    def __init__(self, layers, heads, head_dim, page_size=16, device="cuda"):

        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device

        self.k_pages = [[] for _ in range(layers)]
        self.v_pages = [[] for _ in range(layers)]

    def append(self, layer, k, v):

        self.k_pages[layer].append(k.detach())
        self.v_pages[layer].append(v.detach())

    def get(self, layer):

        if len(self.k_pages[layer]) == 0:
            return None, None

        k = torch.cat(self.k_pages[layer], dim=2)
        v = torch.cat(self.v_pages[layer], dim=2)

        return k, v

    def reset(self):

        self.k_pages = [[] for _ in range(self.layers)]
        self.v_pages = [[] for _ in range(self.layers)]