from src.data_loader import load_product_descriptions
from src.vector_store import build_vector_store
from src.config import settings

if __name__ == "__main__":
    docs = load_product_descriptions("data/prod_small.csv")
    build_vector_store(docs, persist_dir="data/faiss_index")
