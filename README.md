# Gene Chatbot – AI-Powered Product Discovery Assistant

Gene Chatbot simplifies online shopping by turning traditional search into a conversational experience. Instead of keyword-based scrolling through irrelevant products, users can ask natural-language questions and receive personalized, context-aware recommendations.

---

## Problem Statement

Searching for something as simple as a pair of shoes online often leads to endless scrolling, irrelevant results, and decision fatigue. Traditional search engines rely heavily on keyword matching and often fail to understand user intent.

**Gene Chatbot** is an AI-powered assistant designed to solve this problem by understanding what the user truly wants, enabling fast, relevant, and personalized product discovery.

---

## Features

* 🤖 Powered by Large Language Models (LLMs)
* 🔍 Uses Retrieval Augmented Generation (RAG) for contextual answers
* 🛠️ Incorporates multiple specialized agents using ReAct framework
* 🔁 Supports multi-turn conversations with memory and personalization
* 🌐 Enables real-time search via Tavily for up-to-date responses

---

## Architecture Overview

Gene Chatbot uses two key tools in a RAG + ReAct framework:

### 1. `create_retriever_tool`

* Wraps a FAISS-based vector store containing \~50,000 Amazon product descriptions
* Retrieves relevant context for each query, then calls the LLM to generate a response using ReAct logic

### 2. `search_tavily`

* Integrates TavilySearch to fetch real-time web data (e.g., current stock prices, latest research, product trends)
* Helps overcome the LLM's knowledge cutoff by dynamically retrieving updated information

![image](https://github.com/user-attachments/assets/b06c357f-4a0f-401f-bd84-144c106544dc)


### Core Components:

* **LLM Initialization**: OpenAI model (`temperature=0`) for deterministic outputs
* **ReAct Agent**: Performs reasoning, decides whether to call a tool, and responds
* **Agent Executor**: Manages query execution, tool usage, and logs behavior via verbose mode

---

## Cost Estimation & Optimization

Running a GenAI application involves three cost areas:

### Cost Components

* **Prompt Cost**: Based on number and complexity of input queries
* **Generation Cost**: Driven by model output size and response depth
* **Fixed Cost**: Infrastructure, storage, and API access fees

### Cost Optimization Strategy

1. **Minimize Token Usage**: Keep prompts and responses concise
2. **Select Efficient Models**: Use smallest LLM that meets quality needs
3. **Monitor Token Flow**: Use OpenAI callbacks to track actual usage

### Token Measurement Methodology

We use LangChain’s OpenAI callback mechanism (recommended by LangChain) to measure:

* Total tokens used
* Prompt vs. completion tokens
* Cost per query in USD

📖 [LangChain Token Usage Tracking](https://python.langchain.com/v0.1/docs/modules/model_io/llms/token_usage_tracking/)

---

## Benchmarking Token Usage

We ran 10 sample queries through the `agent_executor` to track cost:

| Metric               | Value         |
| -------------------- | ------------- |
| Max Output Tokens    | \~600 tokens  |
| Total Cost (10 runs) | \~\$0.009 USD |

Based on these results, we recommend capping response tokens to **2,000** as a safe upper limit.

### Monitoring Workflow:

1. Define a representative query set
2. Loop through queries using the agent
3. Track token usage and cost per query
4. Store results in a table for analysis
5. Optimize model configuration and token limits accordingly

---

## Visuals

### FAISS Vector Store (Embeddings Storage)

![FAISS Vector](https://python.langchain.com/v0.1/assets/images/vector_stores-125d1675d58cfb46ce9054c9019fea72.jpg)

### Cost Optimization in GenAI

![Cost Diagram](https://deepchecks.com/wp-content/uploads/2024/09/img-cost-optimization-in-generative.jpg)

---

## 👤 Author

**Rajiv Verma**
Senior Engineering Manager | AI/ML Specialist
📍 Atlanta, GA
🔗 [LinkedIn](https://github.com/rajivverma11)) | 🧑‍💻 [GitHub](https://github.com/rajivverma11)]

---

## 📄 License

MIT License
