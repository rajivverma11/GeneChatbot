from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from src.config import settings


def create_agent_executor(tools):
    prompt = hub.pull("hwchase17/react")
    llm = ChatOpenAI(model=settings['MODEL_NAME_4O_MINI'], openai_api_key=settings['OPENAI_API_KEY'])
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
