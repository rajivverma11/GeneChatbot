import os
import gradio as gr

from src.config import settings
from src.vector_store import load_vector_store
from src.agent_executor import create_agent_executor, run_queries_with_cost
from src.memory.summary_agent import run_with_memory, benchmark_memory_conversation, summary_agent_executor
from src.tools import search_tavily, amazon_product_search

# Set up API keys from Hugging Face secrets
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = os.environ["TAVILY_API_KEY"]

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

def run_cost_analysis():
    retriever = load_vector_store("data/faiss_index").as_retriever()
    tools = [search_tavily, amazon_product_search]
    executor = create_agent_executor(tools)
    df_results = run_queries_with_cost(executor, queries_cost)
    return df_results.to_markdown(index=False)

def run_memory_mode():
    responses = []
    for i, query in enumerate(queries_memory):
        res = summary_agent_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": "demo-session"}}
        )
        responses.append(f"Turn {i+1}: {query}\n🧠 {res['output']}")
    return "\n\n".join(responses)

def interface(mode):
    if mode == "cost":
        return run_cost_analysis()
    elif mode == "memory":
        return run_memory_mode()
    else:
        return "Invalid mode"

gr.Interface(
    fn=interface,
    inputs=gr.Dropdown(["cost", "memory"], label="Select Mode"),
    outputs="text",
    title="GenAI Agent Runner"
).launch()
