import torch.nn.functional as F

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def step(self, images, tokens, labels):

        logits = self.model(images, tokens)

        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()