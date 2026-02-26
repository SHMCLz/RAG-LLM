from __future__ import annotations
import requests

class DeepSeekIntranetClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        headers = {"Content-Type":"application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model":"deepseek-r1", "prompt":prompt, "max_tokens": int(max_tokens), "temperature": float(temperature)}
        r = requests.post(f"{self.base_url}/generate", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return str(r.json().get("text",""))

def trim_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()
