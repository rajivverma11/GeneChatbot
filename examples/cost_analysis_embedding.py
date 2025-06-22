from src.data_loader import load_product_descriptions
from src.vector_store import build_vector_store
from src.token_utils import count_tokens, estimate_embedding_cost

# Load product descriptions
product_descriptions = load_product_descriptions("data/sample_dataset.csv")
print(f"Loaded {len(product_descriptions)} product entries.")

# Tokenize each description
total_tokens = 0
model = "text-embedding-3-small"

token_counts = []
for desc in product_descriptions:
    tokens = count_tokens(desc, model)
    token_counts.append(tokens)
    total_tokens += tokens

print(f"📊 Total Tokens: {total_tokens}")
print(f"📈 Cost to embed ({model}): ${estimate_embedding_cost(total_tokens, model)}")

# Scale-up simulation
for scale in [10, 100, 1000, 2000]:
    scaled_tokens = total_tokens * scale
    estimated_cost = estimate_embedding_cost(scaled_tokens, model)
    print(f"If dataset has {len(product_descriptions) * scale} rows → Cost ≈ ${estimated_cost}")
