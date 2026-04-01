import torch
import torch.nn.functional as F
from multimodal.vlm_model import VLM
from evaluation.caption_metrics import bleu_score


# -----------------------------
# Caption evaluation
# -----------------------------

def evaluate_caption(model, dataloader, device="cuda"):

    model.eval()

    scores = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            logits, _, _, _ = model(images, tokens)

            preds = torch.argmax(logits, dim=-1)

            score = bleu_score(preds, tokens)

            scores.append(score)

    if len(scores) == 0:
        return 0

    return sum(scores) / len(scores)


# -----------------------------
# VQA evaluation
# -----------------------------

def evaluate_vqa(model, dataloader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            # dataset returns dictionary
            images = batch["image"].to(device)

            # question tokens
            tokens = batch["tokens"].to(device)

            # Prefer explicit labels/answers from dataloader.
            # Fallback to tokens when labels are unavailable.
            if "labels" in batch:
                answers = batch["labels"].to(device)
            elif "answer" in batch:
                answers = batch["answer"].to(device)
            else:
                answers = batch["tokens"].to(device)

            logits, _, _, _ = model(images, tokens)

            preds = logits.argmax(dim=-1)

            # Evaluate token accuracy only on supervised answer tokens.
            valid = answers != -100
            if valid.any():
                correct += (preds[valid] == answers[valid]).sum().item()
                total += valid.sum().item()

    if total == 0:
        return 0

    return correct / total


# -----------------------------
# Image-text retrieval
# -----------------------------

def evaluate_retrieval(model, dataloader, device):

    import torch
    import torch.nn.functional as F

    model.eval()

    image_embs = []
    text_embs = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            outputs = model(images, tokens)

            # unpack explicitly
            logits = outputs[0]
            img_emb = outputs[1]
            txt_emb = outputs[2]

            # ensure correct shapes
            if img_emb.dim() != 2:
                img_emb = img_emb.mean(dim=1)

            if txt_emb.dim() != 2:
                txt_emb = txt_emb.mean(dim=1)

            image_embs.append(img_emb)
            text_embs.append(txt_emb)

    image_embs = torch.cat(image_embs, dim=0)
    text_embs = torch.cat(text_embs, dim=0)

    # normalize embeddings
    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    similarity = image_embs @ text_embs.T

    preds = similarity.argmax(dim=1)

    labels = torch.arange(len(preds), device=preds.device)

    accuracy = (preds == labels).float().mean().item()

    return accuracy