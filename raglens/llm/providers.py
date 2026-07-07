import os

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODELS = {
    "openai": "gpt-5-nano",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen3:8b",
}


class OpenAILLMProvider:

    def __init__(self, model: str | None = None, api_key: str | None = None):

        from openai import OpenAI

        self.model = model or _DEFAULT_MODELS["openai"]
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text


class GroqLLMProvider:

    def __init__(self, model: str | None = None, api_key: str | None = None):

        from groq import Groq

        self.model = model or _DEFAULT_MODELS["groq"]
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class OllamaLLMProvider:

    def __init__(self, model: str | None = None, base_url: str | None = None):

        from langchain_ollama import ChatOllama

        self.model = model or _DEFAULT_MODELS["ollama"]
        self._chat = ChatOllama(
            model=self.model,
            base_url=base_url or os.getenv("OLLAMA_HOST"),
        )

    def generate(self, prompt: str) -> str:

        return self._chat.invoke(prompt).content


_PROVIDERS = {
    "openai": OpenAILLMProvider,
    "groq": GroqLLMProvider,
    "ollama": OllamaLLMProvider,
}


def get_llm_provider(provider: str, model: str | None = None, **kwargs):

    provider = provider.lower()

    cls = _PROVIDERS.get(provider)

    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Choose from {sorted(_PROVIDERS)}."
        )

    return cls(model=model, **kwargs)
