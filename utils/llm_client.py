import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

def get_llm_client(provider="google"):
    """
    Central factory function for LLM clients.
    Allows easy switching between cloud and local.
    """
    provider = provider.lower()

    match provider:
        case "google" | "gemini":
            GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            return OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

        case "groq":
            GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
            groq_api_key = os.getenv("GROQ_API_KEY")
            return OpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)

        case "local" | "lmstudio":
            LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL")
            LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")
            return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

        case _:
            raise ValueError(
                f"Unknown provider: {provider}. Available: google, groq, local, lmstudio."
            )
