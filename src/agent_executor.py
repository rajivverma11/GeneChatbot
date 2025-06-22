### src/agent_executor.py
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.callbacks import get_openai_callback
from src.config import settings

from src.config import settings
import os

# Set env for libraries that read from env
os.environ["OPENAI_API_KEY"] = settings["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = settings["TAVILY_API_KEY"]

# Use models
model = settings["MODEL_NAME_4O_MINI"]
embedding_model = settings["MODEL_DEFAULT_EMBEDDING"]



def create_agent_executor(tools, model_name= model, max_tokens=2000):
    
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=max_tokens)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10
    )
    return executor


def run_queries_with_cost(agent_executor, queries):
    results = []
    for query in queries:
        with get_openai_callback() as cb:
            agent_executor.agent.stream_runnable = False
            response = agent_executor.invoke({"input": query})
            results.append({
                "Query": query,
                "Output": response.get("output", ""),
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
                "Total Tokens": cb.total_tokens,
                "Total Cost (USD)": cb.total_cost
            })

    df = pd.DataFrame(results)
    print("\n=== TOKEN/COST SUMMARY ===")
    print(df[["Prompt Tokens", "Completion Tokens", "Total Tokens", "Total Cost (USD)"]].describe())
    print(f"\n💰 Total Cost for {len(df)} queries: ${df['Total Cost (USD)'].sum():.4f}")
    return df
