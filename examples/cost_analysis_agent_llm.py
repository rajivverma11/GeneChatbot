import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.callbacks import get_openai_callback
from src.tools import search_tavily, amazon_product_search

# Define query set
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

# Use GPT-4o-mini with 2K output token cap
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2000)
prompt = hub.pull("hwchase17/react")
tools = [search_tavily, amazon_product_search]

# Create agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# Collect results
results = []

for query in queries:
    with get_openai_callback() as cb:
        agent_executor.agent.stream_runnable = False  # Avoid partial output streaming
        response = agent_executor.invoke({"input": query})
        results.append({
            "Query": query,
            "Output": response.get("output", ""),
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Tokens": cb.total_tokens,
            "Total Cost (USD)": cb.total_cost
        })

# Create DataFrame
df = pd.DataFrame(results)

# Display summary
print("\n=== TOKEN/COST SUMMARY ===")
print(df[["Prompt Tokens", "Completion Tokens", "Total Tokens", "Total Cost (USD)"]].describe())
print(f"\n💰 Total Cost for {len(df)} queries: ${df['Total Cost (USD)'].sum():.4f}")

# Save to CSV
df.to_csv("examples/agent_cost_analysis_output.csv", index=False)
