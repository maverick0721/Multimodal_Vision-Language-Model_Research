import torch
from PIL import Image
import torchvision.transforms as T

from multimodal.vlm_model import VLM
from inference.generate import generate

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
        vocab=32000,
        device=None
    ):


        # Device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        # Load VLM
        self.model = VLM(vocab=vocab)

        if checkpoint is not None:

            state = torch.load(
                checkpoint,
                map_location=self.device
            )

            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    
        # Retrieval system
        texts = load_knowledge("knowledge/wiki.txt")

        embedder = SimpleEmbedder()

        self.retriever = SimpleRetriever(
            texts,
            embedder
        )

    
        # Tools + reasoning
        self.router = ToolRouter()
        self.agent = ReActAgent(self.model)

   
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

        image = self.load_image(image)

        # -------- tool detection --------

        tool_result = self.router.run(question)

        if tool_result is not None:

            question = f"""
Tool result: {tool_result}

User question:
{question}
"""

        # -------- build prompt --------

        prompt = self.build_prompt(question)

        # -------- reasoning agent --------

        answer = self.agent.run(
            image=image,
            question=prompt
        )

        # -------- update memory --------

        self.memory.add(question, answer)

        return answer