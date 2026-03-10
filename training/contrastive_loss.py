import torch
import torch.nn.functional as F


def siglip_loss(image_emb, text_emb, temperature=0.07):

    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = image_emb @ text_emb.t() / temperature

    labels = torch.arange(
        image_emb.size(0),
        device=image_emb.device
    )

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_t) / 2