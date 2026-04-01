from dataset.instruction_format import build_prompt


def test_build_prompt_structure_and_strip():
    prompt = build_prompt("What is in this image?")

    assert prompt.startswith("User: What is in this image?")
    assert prompt.endswith("Assistant:")
    assert "\n\n" not in prompt
