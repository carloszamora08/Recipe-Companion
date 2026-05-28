from rag import Assistant
from dotenv import load_dotenv

import os
import subprocess

WELCOME = """
╔══════════════════════════════════════════════════════╗
║                Recipe Companion                      ║
║                                                      ║
║  Tell me about your available ingredients.           ║
║  Type '/clear' to reset conversation history.        ║
║  Type '/exit' to leave.                              ║
╚══════════════════════════════════════════════════════╝
"""


def load_config_from_env() -> dict[str, str | None]:
    """Load raw RAG configuration values from environment variables."""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "top_k": os.getenv("TOP_K")
    }


def main():
    print("Initializing assistant...")
    config = load_config_from_env()
    assistant = Assistant.from_config(config)
    subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)

    print(WELCOME)

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "/exit":
            print("Goodbye!")
            break

        if question.lower() == "/clear":
            assistant.clear_history()
            subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)
            print("\nConversation history cleared.\n")
            continue

        response = assistant.ask(question)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    load_dotenv()
    main()
