# src/memory/summary_agent.py

import time
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain import hub

from src.config import settings
from src.tools import search_tavily, amazon_product_search

tools = [search_tavily, amazon_product_search]

# Load ReAct-style prompt
prompt = hub.pull("hwchase17/react-chat")

# Create streaming LLM for efficiency
summary_llm = ChatOpenAI(
    model=settings["MODEL_NAME_4O_MINI"],
    temperature=0,
    streaming=True,
    api_key=settings["OPENAI_API_KEY"]
)

# Create summarizing memory to reduce token bloat

summary_memory = ConversationSummaryMemory(
    llm=summary_llm,
    memory_key="chat_history"
)

def get_summary_agent_executor():
    global summary_agent_executor
    if summary_agent_executor is None:
        summary_memory = ConversationSummaryMemory(...)
        summary_agent_executor = create_agent_executor_with_memory(summary_memory)
    return summary_agent_executor

# Create ReAct agent with memory support
summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=tools,
    prompt=prompt
)

# Agent executor with memory attached
summary_agent_executor = AgentExecutor(
    agent=summary_react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    memory=summary_memory
)

def run_with_memory(query: str, session_id: str = "test-session"):
    """Runs the agent with memory for a single input."""
    return summary_agent_executor.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )

def benchmark_memory_conversation(queries: list[str], session_id: str = "test-session-2"):
    """Measures time and token cost per query to evaluate memory benefits."""
    from langchain_community.callbacks import get_openai_callback
    import pandas as pd
    import time

    results = []
    times = []

    for query in queries:
        start = time.time()
        with get_openai_callback() as cb:
            response = summary_agent_executor.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
        duration = time.time() - start
        times.append(duration)

        results.append({
            "Query": query,
            "Response": response,
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost,
            "Time (sec)": duration
        })

    df = pd.DataFrame(results)
    print("\n=== TOKEN/COST SUMMARY ===")
    print(df[["Prompt Tokens", "Completion Tokens", "Total Tokens", "Total Cost (USD)"]].describe())
    total_cost = df["Total Cost (USD)"].sum()
    avg_time = sum(times) / len(times) if times else 0

    return results, df, total_cost, avg_time

