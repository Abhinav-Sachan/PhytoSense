import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class PlantLLM:

    def __init__(self):
        # Initialize Groq using the API key from .env
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please check the backend/.env file."
            )

        self.client = Groq(api_key=api_key)

        # Use the current Groq model directly.
        # This prevents an old LLM_MODEL value in .env
        # from selecting a decommissioned model.
        self.model = "openai/gpt-oss-20b"

    def generate_status(self, metrics):

        prompt = (
            "You are an AI plant monitoring assistant called PhytoSense. "
            "Analyze the following plant bio-electric measurements.\n\n"
            f"Plant state: {metrics['label']}\n"
            f"Confidence: {metrics['confidence']:.0%}\n"
            f"RMS amplitude: {metrics['features']['rms']:.2f}\n"
            f"Dominant frequency: "
            f"{metrics['features']['dominant_frequency']:.1f} Hz\n"
            f"Zero-crossing rate: "
            f"{metrics['features']['zero_crossing_rate']:.3f}\n"
            f"Peak amplitude: "
            f"{metrics['features']['max_amplitude']:.2f}\n\n"
            "Provide a concise 2-sentence plain-language diagnosis. "
            "Do not claim that the sensor prediction is a scientifically "
            "proven diagnosis."
        )

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,
            temperature=0.3
        )

        return res.choices[0].message.content

    def generate_chat_reply(self, message, history, metrics):

        system_context = (
            "You are PhytoSense, an AI assistant for a plant "
            "bio-electric monitoring system. "
            "Help the user understand the plant's sensor readings "
            "in simple and clear language. "
            "Do not present sensor predictions as scientifically "
            "proven diagnoses.\n\n"
            f"Current plant state: {metrics.get('label', 'Unknown')}\n"
            f"Current confidence: "
            f"{metrics.get('confidence', 0):.0%}\n"
            f"Current features: {metrics.get('features', {})}"
        )

        formatted_history = [
            {
                "role": "system",
                "content": system_context
            }
        ]

        # Add previous conversation messages
        for msg in history:
            if "role" in msg and "content" in msg:
                formatted_history.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )

        # Add the current user message
        formatted_history.append(
            {
                "role": "user",
                "content": message
            }
        )

        res = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_history,
            max_tokens=300,
            temperature=0.7
        )

        return res.choices[0].message.content