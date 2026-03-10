import torch


class PagedKVCache:

    def __init__(self, layers, heads, head_dim, page_size=16, max_pages=1024):

        self.page_size = page_size

        self.k = torch.zeros(
            layers, max_pages, heads, page_size, head_dim,
            device="cuda"
        )

        self.v = torch.zeros(
            layers, max_pages, heads, page_size, head_dim,
            device="cuda"
        )

        self.next_page = 0

    def allocate_page(self):

        page = self.next_page
        self.next_page += 1
        return page

    def write(self, layer, page, position, k, v):

        self.k[layer, page, :, position] = k
        self.v[layer, page, :, position] = v

    def read(self, layer, pages):

        keys = self.k[layer, pages].reshape(-1, *self.k.shape[2:])
        values = self.v[layer, pages].reshape(-1, *self.v.shape[2:])

        return keys, values