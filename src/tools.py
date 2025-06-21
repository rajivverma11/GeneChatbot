from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from src.vector_store import load_vector_store

def truncate_text(text: str, max_chars: int = 3000) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


@tool
def amazon_product_search(query: str):
    '''Search for information about Amazon products.
    For any questions related to Amazon products, this tool must be used.'''

    retriever = load_vector_store("data/faiss_index").as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,  # You will inject this in main
        name="amazon_search",
        description="Search for Amazon products"
    )
    return retriever_tool.invoke(query)

@tool
def search_tavily(query: str):
    ''' Executes a web search using the TavilySearchResults tool.

    Parameters:
        query (str): The search query entered by the user.

    Returns:
        list: A list of search results containing answers, raw content, and images.'''
    search_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    result = search_tool.invoke(query)
    if isinstance(result, str):
        return truncate_text(result)
    elif isinstance(result, list):
        return truncate_text("\n".join([str(r) for r in result]))
    else:
        return str(result)


