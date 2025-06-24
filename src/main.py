# src/main.py

import os
import argparse
from src.config import settings
from src.vector_store import load_vector_store
from src.agent_executor import create_agent_executor, run_queries_with_cost
from src.tools import search_tavily, amazon_product_search
from memory.summary_agent import summary_agent_executor, run_with_memory, benchmark_memory_conversation


# Set environment variables
os.environ["OPENAI_API_KEY"] = settings["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = settings["TAVILY_API_KEY"]

# CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cost", "memory", "benchmark_memory", "run_memory"], default="cost", help="Execution mode")
args = parser.parse_args()

# Shared values
model = settings["MODEL_NAME_4O_MINI"]
embedding_model = settings["MODEL_DEFAULT_EMBEDDING"]
tools = [search_tavily, amazon_product_search]
queries_memory = [
    "What are the best shoes on Amazon?",
    "Can you find something under $100?",
    "Show me a running shoe option."
]
queries_cost = [
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

if __name__ == "__main__":
    if args.mode == "cost":
        # 🔁 Batch cost-tracking mode
        retriever = load_vector_store("data/faiss_index").as_retriever()
        executor = create_agent_executor(tools)
        df_results = run_queries_with_cost(executor, queries_cost)
        df_results.to_csv("examples/agent_cost_analysis_output.csv", index=False)
        print("Cost and token usage saved to examples/agent_cost_analysis_output.csv")

    elif args.mode == "memory":
        # 🧠 Basic memory mode (single multi-turn example)
        print("\nRunning memory-enabled multi-turn session:\n")
        for i, query in enumerate(queries_memory):
            print(f"\nTurn {i+1} | Query: {query}")
            res = summary_agent_executor.invoke(
                {"input": query},
                config={"configurable": {"session_id": "demo-session"}}
            )

    elif args.mode == "benchmark_memory":
        # 📊 Run memory benchmark mode
        print("\nBenchmarking memory-based multi-turn conversation with timing analysis:\n")
        benchmark_memory_conversation(queries_memory)

    else:
        # ▶️ Run memory demo function
        print("\nRunning demo memory session from utility method:\n")
        run_with_memory(queries_memory[0])

