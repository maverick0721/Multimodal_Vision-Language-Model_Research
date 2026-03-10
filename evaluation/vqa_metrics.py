def vqa_accuracy(pred, answer):

    pred = pred.strip().lower()
    answer = answer.strip().lower()

    return float(pred == answer)