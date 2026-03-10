import torch

from multimodal.vlm_model import VLM
from evaluation.caption_metrics import bleu_score


def evaluate_caption(model, dataset):

    scores = []

    for image, question, answer in dataset:

        pred = model.generate(image, question)

        score = bleu_score(answer, pred)

        scores.append(score)

    return sum(scores) / len(scores)