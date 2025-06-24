import os
import argparse
import gradio as gr
from src.config import settings
from src.vector_store import load_vector_store
from src.agent_executor import create_agent_executor, run_queries_with_cost
from src.tools import search_tavily, amazon_product_search
from src.memory.summary_agent import run_with_memory, benchmark_memory_conversation, summary_agent_executor

# Set environment variables
os.environ["OPENAI_API_KEY"] = settings["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = settings["TAVILY_API_KEY"]

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
# Use models
model = settings["MODEL_NAME_4O_MINI"]
embedding_model = settings["MODEL_DEFAULT_EMBEDDING"]

def run_cost_analysis():
    retriever = load_vector_store("data/faiss_index").as_retriever()
    tools = [search_tavily, amazon_product_search]
    
    executor = create_agent_executor(tools)
    df_results = run_queries_with_cost(executor, queries_cost)
    df_results.to_csv("examples/agent_cost_analysis_output.csv", index=False)
    return "Cost and token usage saved to examples/agent_cost_analysis_output.csv"

def run_memory_mode():
    queries = [
        "What are the best shoes on Amazon?",
        "Can you find something under $100?",
        "Show me a running shoe option."
    ]
    responses = []
    for i, query in enumerate(queries_memory):
        res = summary_agent_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": "demo-session"}}
        )
        responses.append((f"Turn {i+1}: {query}", res["output"]))
    return responses

def run_benchmark_memory():
    results, df, total_cost, avg_time = benchmark_memory_conversation(queries_memory)
    df.to_csv("examples/memory_benchmark_output.csv", index=False)
    return f"Benchmark complete. Total cost: ${total_cost:.5f}, Avg response time: {avg_time:.2f} sec"

def run_custom_memory():
    run_with_memory(queries_memory[0])
    return "Custom memory demo completed."

def gradio_ui():
    def interface(mode):
        if mode == "cost":
            return run_cost_analysis()
        elif mode == "memory":
            return run_memory_mode()
        elif mode == "benchmark_memory":
            return run_benchmark_memory()
        elif mode == "run_memory":
            return run_custom_memory()
        else:
            return "Invalid mode selected."

    gr.Interface(
        fn=interface,
        inputs=gr.Dropdown(["cost", "memory", "benchmark_memory", "run_memory"], label="Select Mode"),
        outputs="text",
        title="GenAI Agent Runner"
    ).launch()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cost", help="Mode to run: cost, memory, benchmark_memory, run_memory, or gradio")
    args = parser.parse_args()

    if args.mode == "cost":
        print(run_cost_analysis())
    elif args.mode == "memory":
        print("\n Running memory-enabled multi-turn session:\n") 
        for i, query in enumerate([
            "What are the best shoes on Amazon?",
            "Can you find something under $100?",
            "Show me a running shoe option."
        ]):
            print(f"\n Turn {i+1} | Query: {query}") 
            res = summary_agent_executor.invoke(
                {"input": query},
                config={"configurable": {"session_id": "demo-session"}}
            )
            print("🧠 Response:", res["output"])
    elif args.mode == "benchmark_memory":
        print(run_benchmark_memory())
    elif args.mode == "run_memory":
        print(run_custom_memory())
    elif args.mode == "gradio":
        gradio_ui()
    else:
        print("Invalid mode. Use one of: cost, memory, benchmark_memory, run_memory, gradio")

if __name__ == "__main__":
    main()
