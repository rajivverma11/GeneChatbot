### src/main.py
from src.config import settings
from src.vector_store import load_vector_store
from src.agent_executor import create_agent_executor, run_queries_with_cost
from src.tools import search_tavily, amazon_product_search

from src.config import settings
import os

# Set env for libraries that read from env
os.environ["OPENAI_API_KEY"] = settings["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = settings["TAVILY_API_KEY"]

# Use models
model = settings["MODEL_NAME_4O_MINI"]
embedding_model = settings["MODEL_DEFAULT_EMBEDDING"]

if __name__ == "__main__":
    # Load retriever from saved FAISS index
    retriever = load_vector_store("data/faiss_index").as_retriever()

    # Initialize tools
    tools = [search_tavily, amazon_product_search]

    # Sample queries for agent evaluation
    queries = [
        "Summarize the history of artificial intelligence",
        "What is the weather in London for next week?",
        "What are the best shoes available in Amazon?",
        "What are some top clothing brands in Amazon?",
        "When are the next Manchester United matches scheduled?",
        "What kind of clothes should we use for travel in Switzerland next week?",
        "How to install Backsplash Wallpaper? Also find some brands on Amazon",
        "What is the current weather in Toronto?",
        "Give me a detailed summary of wedding dresses that I can buy in Amazon"
    ]

    # Create agent and run queries with cost tracking
    executor = create_agent_executor(tools)
    df_results = run_queries_with_cost(executor, queries)

    # Save results to CSV
    df_results.to_csv("examples/agent_cost_analysis_output.csv", index=False)
    print("\nCost and token usage saved to examples/agent_cost_analysis_output.csv")
