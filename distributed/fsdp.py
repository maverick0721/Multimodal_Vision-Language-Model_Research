import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def wrap_model(model):

    model = FSDP(
        model,
        device_id=torch.cuda.current_device()
    )

    return model