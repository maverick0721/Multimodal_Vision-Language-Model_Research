def build_prompt(question):

    prompt = f"""
User: {question}
Assistant:
"""

    return prompt.strip()