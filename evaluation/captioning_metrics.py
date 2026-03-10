def bleu(pred,ref):

    pred = pred.split()
    ref = ref.split()

    match = sum(
        1 for w in pred if w in ref
    )

    return match / len(pred)