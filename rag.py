# You might need the following imports. Feel free to change it if you opt for different libraries.
from __future__ import annotations

import os
import pandas as pd
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from openai import OpenAI

# Default configs
DEFAULT_DATA_DIR = "data/Recipes from around the world.csv"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 4
DEFAULT_TOP_N = 3


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model", DEFAULT_LLM_MODEL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "rerank_model": config.get("rerank_model", DEFAULT_RERANK_MODEL),  
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k", DEFAULT_TOP_K),
        ),
        "top_n": _parse_int_setting(
            "TOP_N",
            config.get("top_n", DEFAULT_TOP_N)
        )
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["top_n"] <= 0:  
        raise ValueError("TOP_N must be > 0")

    return resolved


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    
    df = pd.read_csv(data_dir, encoding="latin-1")

    documents = []
    for _, row in df.iterrows():
        page_content = f"""Recipe: {row.get('recipe_name', 'Unknown')}
        Ingredients: {row.get('ingredients', 'N/A')}
        Cooking Time: {row.get('cooking_time_minutes', 'N/A')} minutes
        Prep Time: {row.get('prep_time_minutes', 'N/A')} minutes
        Dietary Restrictions: {row.get('dietary_restrictions', 'None')}"""

        metadata = {
            "cuisine":              str(row.get('cuisine', 'Unknown')),
            "servings":             str(row.get('servings', 'N/A')),
            "calories_per_serving": str(row.get('calories_per_serving', 'N/A')),
            "recipe_name":          str(row.get('recipe_name', 'Unknown')),
            "document_type":        "recipe",
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"Loaded {len(documents)} recipes from {data_dir}")
    return documents


def build_index(
        recipes: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    embeddings = embedding_model.encode([recipe.page_content for recipe in recipes], normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        recipes: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:    
    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_embedding, k)

    results = []
    for (score, idx) in zip(scores[0], indices[0]):
        results.append({
            "text": recipes[idx].page_content,
            "score": score,
            "metadata": recipes[idx].metadata
        })

    return results


def rerank(query: str, results: list[dict], reranker: CrossEncoder, top_n: int = DEFAULT_TOP_N) -> list[dict]:
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, results), key=lambda x: x[0], reverse=True
    )
    return [r for _, r in ranked[:top_n]]


SYSTEM_PROMPT = """You are a recipe assistant. Your ONLY knowledge comes from the recipe database provided in the context. Follow these rules:
                - Recommend the best matching recipe based on the ingredients the user mentions.
                - ONLY recommend recipes that exist in the provided context. Never invent or suggest recipes outside of it.
                - When suggesting a recipe, always include: name, cuisine, ingredients, cooking time, prep time, servings, calories, and dietary restrictions.
                - Present the recipe in a clear, step-by-step format, using your own capabilities to write the cooking instructions in a friendly and easy-to-follow way.
                - If no recipe matches the user's ingredients, say "I couldn't find a matching recipe in my database for those ingredients."
                - Do not use any prior culinary knowledge outside of the context."""


class Assistant:
    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            recipes: list[Document],
            client: OpenAI,
            reranker: CrossEncoder,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.recipes = recipes
        self.client = client
        self.reranker = reranker
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.top_n = self.config["top_n"]
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None, n: int | None = None) -> str:        
        k = k or self.top_k
        n = n or self.top_n
        if n > k:
            raise ValueError(f"top_n ({n}) cannot be greater than top_k ({k})")

        results = retrieve(question, self.index, self.model, self.recipes, k)

        reranked_results = rerank(question, results, self.reranker, top_n=n)

        context = "\n\n---\n\n".join(
            f"[{r['metadata']['recipe_name']}]\n{r['text']}"
            for r in reranked_results
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages
        )

        reply = response.choices[0].message.content
        self.history.append({'role': 'user', 'content': question})
        self.history.append({'role': 'assistant', 'content': reply})

        return reply
 

    def clear_history(self) -> None:
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Assistant:
        resolved_config = resolve_config(config)

        print("Loading documents...")
        recipes = load_documents()
        print(f"  Loaded {len(recipes)} documents")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(recipes, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        reranker = CrossEncoder(resolved_config["rerank_model"])

        print("Ready!\n")
        return cls(index, embedding_model, recipes, client, reranker, resolved_config)
