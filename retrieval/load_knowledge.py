def load_knowledge(path):

    texts = []

    with open(path) as f:

        for line in f:

            texts.append(line.strip())

    return texts