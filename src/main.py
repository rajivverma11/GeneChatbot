from src.config import settings
from src.data_loader import load_product_descriptions
from src.vector_store import build_vector_store
from src.agent_executor import create_agent_executor
from src.tools import amazon_product_search, search_tavily
from src.vector_store import load_vector_store



def main():
    # Inject retriever into tools (optional, you may refactor the tools to accept retriever)
    tools = [search_tavily, amazon_product_search]
    
    executor = create_agent_executor(tools)

    queries = [
        "What is the best shoes I can find on Amazon?",
        "What is the current weather in Toronto?",
        "How to install Backsplash Wallpaper? Also find some brands on Amazon"
    ]
    for query in queries:
        result = executor.invoke({"input": query})
        print(result)

if __name__ == "__main__":
    main()
