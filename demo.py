import torch
from multimodal.vlm_model import VLM
from inference.generate import generate


model = VLM(vocab=32000)

model.load_state_dict(
    torch.load("experiments/run_001/checkpoint_1000.pt")
)

model.eval()

image = torch.randn(1,3,224,224)

prompt = "Describe this image"

output = generate(model, image, prompt)

print(output)