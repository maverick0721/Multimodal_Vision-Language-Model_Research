import torch

from multimodal.vlm_model import VLM
from experiments.logger import Logger
from text.lora import apply_lora
import torch.distributed as dist
from training.contrastive_loss import siglip_loss

dist.init_process_group("nccl")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


model = VLM(vocab=32000).cuda()

# apply LoRA to decoder
apply_lora(model.text)

# freeze base weights
for name, p in model.named_parameters():

    if "lora_" not in name:

        p.requires_grad = False


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

logger = Logger()


for step in range(10000):

    images = torch.randn(
        8,3,224,224
    ).to(device)

    tokens = torch.randint(
        0,32000,
        (8,128)
    ).to(device)

    logits, img_emb, txt_emb, moe_loss = model(images, tokens)

    # ----- caption loss -----
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    caption_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # ----- contrastive alignment -----
    contrast_loss = siglip_loss(img_emb, txt_emb)

    # ----- final loss -----
    loss = caption_loss + 0.1 * contrast_loss + 0.01 * moe_loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    logger.log(step, loss.item())

    print(
        f"step {step} | total {loss.item():.4f} | "
        f"caption {caption_loss.item():.4f} | "
        f"contrast {contrast_loss.item():.4f}"
    )