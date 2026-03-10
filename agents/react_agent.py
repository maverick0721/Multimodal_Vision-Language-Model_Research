from agents.router import ToolRouter
from inference.generate import generate


class ReActAgent:

    def __init__(self, model):

        self.model = model
        self.router = ToolRouter()

    def run(self, image, question, steps=3):

        history = question

        for _ in range(steps):

            prompt = f"""
You are a multimodal reasoning assistant.

Conversation so far:
{history}

Respond in this format:

Thought:
Action:
"""

            output = generate(self.model, image, prompt)

            tool_result = self.router.run(output)

            if tool_result is None:

                history += f"\nModel: {output}"
                break

            history += f"""
Model: {output}
Observation: {tool_result}
"""

        final_prompt = f"""
{history}

Final Answer:
"""

        answer = generate(self.model, image, final_prompt)

        return answer