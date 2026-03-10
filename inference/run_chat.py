from inference.chat_vlm import VLMChat

chat = VLMChat()

while True:

    image = input("image: ")

    prompt = input("prompt: ")

    out = chat.generate(image, prompt)

    print(out)