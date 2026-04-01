import torch
from PIL import Image
import torchvision.transforms as T

from inference.generate import Generator
from utils.config import load_config

# Retrieval
from retrieval.load_knowledge import load_knowledge
from retrieval.retriever import SimpleRetriever
from retrieval.embedder import SimpleEmbedder

# Tools + reasoning
from agents.router import ToolRouter
from agents.react_agent import ReActAgent

# Memory
from agents.memory import ConversationMemory


class VLMChat:

    def __init__(
        self,
        checkpoint=None,
        vocab=None,
        device=None
    ):


        # Device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        # Load VLM via Generator
        self.generator = Generator(
            checkpoint=checkpoint,
            vocab=vocab,
            device=self.device
        )
        self.model = self.generator.model

    
        # Retrieval system
        texts = load_knowledge("knowledge/wiki.txt")

        embedder = SimpleEmbedder()

        self.retriever = SimpleRetriever(
            texts,
            embedder
        )

    
        # Tools + reasoning
        self.router = ToolRouter()
        self.agent = ReActAgent(self.generator)

   
        # Conversation memory
        self.memory = ConversationMemory(max_turns=5)

   
        # Image transform
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])


    # Load image
    def load_image(self, image):

        if isinstance(image, str):

            img = Image.open(image).convert("RGB")

            image = self.transform(img).unsqueeze(0)

        return image.to(self.device)

   
    # Build retrieval prompt
    def build_prompt(self, question):

        docs = self.retriever.search(question, k=3)

        context = "\n".join(docs)

        history = self.memory.get_context()

        prompt = f"""
Conversation history:
{history}

Retrieved knowledge:
{context}

User question:
{question}

Answer:
"""

        return prompt

   
    # Chat interface
    def chat(self, image, question):

        # Build a grounded prompt with retrieval and short conversation history.
        prompt = self.build_prompt(question)

        # If the question appears tool-appropriate, let the ReAct loop use tools.
        # Otherwise use the standard grounded generation path.
        tool_name = self.router.detect_tool(question)
        if tool_name is not None:
            answer = self.agent.run(image, prompt)
        else:
            answer = self.generator.generate(
                image_path=image,
                prompt=prompt
            )

        # -------- update memory --------

        self.memory.add(question, answer)

        return answer