import math


def calculator(expression):

    try:
        return str(eval(expression))

    except Exception:

        return "calculation_error"


def image_caption_stub(image):

    return "This appears to be an image."


TOOLS = {
    "calculator": calculator,
    "caption": image_caption_stub
}