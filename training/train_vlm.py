import torch

from multimodal.vlm_model import VLM
from experiments.logger import Logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = VLM(vocab=32000).cuda()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

logger = Logger()

for step in range(10000):

    images = torch.randn(
        8,3,224,224
    ).cuda()

    tokens = torch.randint(
        0,32000,
        (8,128)
    ).cuda()

    logits = model(images, tokens)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    logger.log(step,loss.item())

    print(step,loss.item())