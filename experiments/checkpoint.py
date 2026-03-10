import torch

def save_checkpoint(model,optimizer,step,path):

    torch.save({
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "step":step
    },path)


def load_checkpoint(model,optimizer,path):

    ckpt = torch.load(path)

    model.load_state_dict(
        ckpt["model"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    return ckpt["step"]