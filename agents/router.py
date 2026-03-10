import re
from agents.tools import TOOLS


class ToolRouter:

    def __init__(self):

        self.tools = TOOLS

    def detect_tool(self, text):

        if "calculate" in text.lower():

            return "calculator"

        if "describe image" in text.lower():

            return "caption"

        return None

    def run(self, text):

        tool = self.detect_tool(text)

        if tool is None:

            return None

        if tool == "calculator":

            expr = re.findall(r"[0-9\+\-\*\/\.]+", text)

            if len(expr) > 0:

                return self.tools[tool](expr[0])

        return self.tools[tool](text)