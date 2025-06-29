import os
from dotenv import load_dotenv


load_dotenv()

settings = {
    'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
    'TAVILY_API_KEY': os.getenv("TAVILY_API_KEY"),
    'MODEL_NAME': 'gpt-3.5-turbo',
    'MODEL_NAME_4O_MINI': 'gpt-4o',
    'MODEL_DEFAULT_EMBEDDING': 'text-embedding-ada-002'
}



