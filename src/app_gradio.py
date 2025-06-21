import gradio as gr
from src.config import settings
from src.data_loader import load_product_descriptions
from src.vector_store import build_vector_store
from src.agent_executor import create_agent_executor
from src.tools import amazon_product_search, search_tavily



# If needed, update tools to accept this retriever
tools = [search_tavily, amazon_product_search]
agent_executor = create_agent_executor(tools)

# Gradio function
def answer_question(message, history):
    try:
        response = agent_executor.invoke({"input": message})
        return response.get("output", "No output from agent.")
    except Exception as e:
        return f"Error: {str(e)}"

# Launch Gradio chat interface
demo = gr.ChatInterface(
    fn=answer_question,
    title="Product & Web Assistant",
    theme="soft",
    examples=[
        "What are the best running shoes on Amazon?",
        "What is the current weather in Boston?",
        "Give me popular smartwatch brands under $200"
    ]
)

if __name__ == "__main__":
    demo.launch()
