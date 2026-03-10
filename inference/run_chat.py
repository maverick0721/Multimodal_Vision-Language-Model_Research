from inference.chat_vlm import VLMChat


chat = VLMChat(
    checkpoint="experiments/run_001/checkpoint_1000.pt"
)

response = chat.chat(
    image="data/images/dog.jpg",
    question="What animal is this?"
)

print(response)