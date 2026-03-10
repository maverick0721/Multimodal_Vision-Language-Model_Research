import nltk
from nltk.translate.bleu_score import sentence_bleu


def bleu_score(reference, prediction):

    reference = [reference.split()]
    prediction = prediction.split()

    return sentence_bleu(reference, prediction)