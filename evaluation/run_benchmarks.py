import torch

from multimodal.vlm_model import VLM
from evaluation.evaluate import evaluate_caption
from evaluation.evaluate import evaluate_vqa
from evaluation.evaluate import evaluate_retrieval


device = "cuda" if torch.cuda.is_available() else "cpu"

model = VLM(vocab=32000)
model.to(device)


print("Running benchmarks...")

caption_score = evaluate_caption(model)
vqa_score = evaluate_vqa(model)
retrieval_score = evaluate_retrieval(model)


print("Results")

print("Caption BLEU:", caption_score)
print("VQA Accuracy:", vqa_score)
print("Retrieval Recall:", retrieval_score)