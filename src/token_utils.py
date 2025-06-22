import tiktoken

# Maps model → cost per 1K tokens (input only for embeddings)
EMBEDDING_COST_PER_1K = {
    "text-embedding-ada-002": 0.0001,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013
}

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_embedding_cost(tokens: int, model: str = "text-embedding-3-small") -> float:
    cost_per_1k = EMBEDDING_COST_PER_1K.get(model, 0.00002)
    return round((tokens / 1000) * cost_per_1k, 6)

def truncate_text(text: str, max_chars: int = 3000) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text