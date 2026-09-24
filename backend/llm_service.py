import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class PlantLLM:
    def __init__(self):
        # Initializes the Groq client using the key from your .env
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama3-8b-8192")

    def generate_status(self, metrics):
        prompt = (
            f"Analyze plant metrics: {metrics['label']} ({metrics['confidence']:.0%} confidence). "
            f"Features: RMS {metrics['features']['rms']:.2f}, Freq {metrics['features']['dominant_frequency']:.1f}Hz. "
            f"Provide a 2-sentence plain-language diagnosis."
        )
        
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return res.choices[0].message.content

    def generate_chat_reply(self, message, history, metrics):
        # Groq handles system instructions via a dedicated 'system' role at the start of the array
        system_context = f"You are PhytoSense. Current plant state: {metrics['label']}. Features: {metrics['features']}."
        
        formatted_history = [{"role": "system", "content": system_context}]
        
        # Append prior chat context
        for msg in history:
            formatted_history.append({"role": msg["role"], "content": msg["content"]})
            
        # Append the new user query
        formatted_history.append({"role": "user", "content": message})
        
        res = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_history,
            max_tokens=300,
            temperature=0.7
        )
        return res.choices[0].message.content