# LangChain

**LangChain** is a framework for developing applications powered by **large language models (LLMs)**. It simplifies building pipelines that integrate LLMs with external data, memory, and logic to create intelligent applications like chatbots, summarizers, and question-answering systems.

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Basic Concepts](#basic-concepts)
* [Data Handling](#data-handling)
* [Typical Workflow](#typical-workflow)
* [Common Components](#common-components)
* [Common Commands / Code Patterns](#common-commands--code-patterns)

## Introduction

LangChain allows you to:

* Build LLM-powered applications efficiently.
* Combine LLMs with **external data sources**, APIs, and tools.
* Maintain conversation **memory** across interactions.
* Structure prompts and LLM interactions in a modular and reusable way.
* Load, split, embed, and store documents for retrieval-based tasks.

It works with multiple LLM providers (OpenAI, Hugging Face, Cohere, etc.) and supports both local and cloud deployments.

## Installation

### Prerequisites

* Python 3.9+
* Access to an LLM provider (e.g., OpenAI API key)

### Install LangChain

```bash
pip install langchain
```

Optional extras for embeddings and vector stores:

```bash
pip install langchain[openai]  # OpenAI integration
pip install langchain[faiss]   # FAISS vector store
```

## Basic Concepts

* **LLM (Large Language Model)**: Generates text responses.
* **PromptTemplate**: Templates for dynamically structured prompts.
* **Chains**: Sequences of LLM or tool calls to perform a task.
* **Agents**: Systems that can decide which tools or APIs to use to answer a query.
* **Memory**: Tracks conversation or application state.
* **Tools**: External functions, APIs, or scripts that an agent can invoke.
* **Document / Dataset**: Text data used for retrieval or embeddings.
* **Vector Store**: Storage of embeddings for similarity search and retrieval.

## Data Handling: Load, Split, Embed, and Store

LangChain simplifies working with large document collections for retrieval-based tasks:

1. **Load Data**

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("my_docs.txt")
documents = loader.load()
```

Supports many formats: TXT, PDF, CSV, JSON, and even web scraping.

2. **Split Data**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

This ensures embeddings are generated on manageable chunks for better retrieval.

3. **Embed Data**

```python
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in chunks]
```

4. **Store Data in Vector Store**

```python
from langchain.vectorstores import FAISS

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
```

Later, you can **load the vector store** for retrieval:

```python
vector_store = FAISS.load_local("faiss_index", embeddings)
```

5. **Querying Embedded Data**

```python
query = "What is the main topic of the documents?"
results = vector_store.similarity_search(query)
print(results[0].page_content)
```

This workflow enables **retrieval-augmented generation (RAG)**, combining LLMs with embedded document knowledge.

## Typical Workflow

1. Initialize an LLM.
2. Load and split documents.
3. Generate embeddings and store in a vector store.
4. Build a retrieval chain or agent using the vector store.
5. Run queries with context retrieved from the documents.

## Common Components

| Component        | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `LLM`            | Large language model used for generating text                   |
| `PromptTemplate` | Template to structure input to the LLM                          |
| `Chain`          | Sequence of operations using LLMs and tools                     |
| `Agent`          | Intelligent system that chooses tools/actions to complete tasks |
| `Memory`         | Stores conversation or application state                        |
| `Tool`           | External function, API, or script callable by an agent          |
| `Document`       | Text data loaded for retrieval                                  |
| `Vector Store`   | Stores embeddings for similarity search                         |

## Common Commands / Code Patterns

| Task                    | Example                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| Initialize LLM          | `llm = ChatOpenAI(temperature=0.7)`                                                                             |
| Load documents          | `loader = TextLoader("docs.txt"); docs = loader.load()`                                                         |
| Split documents         | `splitter = RecursiveCharacterTextSplitter(chunk_size=500); chunks = splitter.split_documents(docs)`            |
| Embed documents         | `embeddings = OpenAIEmbeddings(); vector_embeddings = [embeddings.embed_query(c.page_content) for c in chunks]` |
| Store vector embeddings | `vector_store = FAISS.from_documents(chunks, embeddings); vector_store.save_local("faiss_index")`               |
| Query vector store      | `results = vector_store.similarity_search("query text")`                                                        |
| Run chain               | `chain = LLMChain(llm=llm, prompt=prompt); chain.run("Hello")`                                                  |
| Initialize agent        | `agent = initialize_agent(tools, llm, agent="zero-shot-react-description")`                                     |

# Ollama vs OpenAI

## 🔹 Ollama
- **What it is**: A tool for running large language models (LLMs) **locally** on your machine.
- **Key features**:
  - Runs models like **LLaMA, Mistral, Gemma**, etc.
  - Easy setup: `ollama run model-name`
  - Works offline (no API calls required).
  - Useful for **privacy-focused** or **air-gapped environments**.
- **Use cases**:
  - Experimenting with open-source LLMs.
  - Running models on your own hardware.
  - Custom fine-tuning and prompt engineering locally.

## 🔹 OpenAI
- **What it is**: A **cloud-based AI platform** providing access to proprietary models via API (e.g., **GPT-4, GPT-4o, DALL·E, Whisper**).
- **Key features**:
  - High performance and state-of-the-art accuracy.
  - Scalable API for apps, chatbots, analysis, and automation.
  - Wide ecosystem: ChatGPT, Playground, Assistants API.
- **Use cases**:
  - Production-grade applications needing reliable, high-quality AI.
  - Natural language understanding, coding assistants, image generation.
  - Businesses requiring **enterprise support**, compliance, and monitoring.

## ⚖️ Summary
- **Ollama** → Best if you want **local, private, and open-source LLMs**.
- **OpenAI** → Best if you need **cutting-edge, cloud-hosted AI with strong support and scalability**.


Here’s a **Markdown note** on Hugging Face **Transformers** and **Embeddings**, similar to the Ollama vs OpenAI one I gave you:

````markdown
# Hugging Face: Transformers & Embeddings

## 🔹 Transformers
- **What it is**: An open-source library by Hugging Face providing **state-of-the-art pre-trained models** for:
  - Natural Language Processing (NLP)
  - Computer Vision (CV)
  - Audio and Multimodal tasks
- **Key features**:
  - Access to thousands of models from the Hugging Face Hub.
  - Supports multiple frameworks: **PyTorch, TensorFlow, JAX**.
  - Simple, high-level API for tasks like text classification, summarization, translation, and question answering.
- **Example**:
  ```python
  from transformers import pipeline

  summarizer = pipeline("summarization")
  print(summarizer("Transformers makes AI super accessible!", max_length=20))
````

## 🔹 Embeddings

* **What they are**: Numerical vector representations of text, images, or audio that capture **semantic meaning**.
* **Why important**:

  * Power search engines (semantic search).
  * Enable clustering, similarity, and recommendation systems.
  * Foundation for Retrieval-Augmented Generation (RAG).
* **Key models**:

  * `all-MiniLM-L6-v2` (lightweight, widely used for semantic search).
  * `sentence-transformers` family for high-quality embeddings.
* **Example (text embeddings)**:

  ```python
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = model.encode(["Hello world", "Hi there"])
  print(embeddings.shape)  # (2, 384)
  ```

## ⚖️ Summary

* **Transformers** → Provide access to **pre-trained models** for a wide range of AI tasks.
* **Embeddings** → Represent inputs as vectors for **semantic understanding**, powering similarity search and RAG.

# Vector Stores

## 🔹 What is a Vector Store?
A **vector store** is a specialized database optimized to store and search **vector embeddings**.  
Instead of looking for exact matches (like SQL), vector stores find **similar items** based on **distance/similarity metrics** (cosine similarity, dot product, Euclidean distance).

## 🔹 Why Use Them?
- Power **semantic search** (find documents with similar meaning, not just keywords).
- Enable **Retrieval-Augmented Generation (RAG)** by feeding relevant context into LLMs.
- Efficiently handle **high-dimensional embeddings** (hundreds or thousands of dimensions).

## 🔹 Popular Vector Stores
- **Open-source / self-hosted**:
  - [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search, lightweight, local use).
  - [Chroma](https://www.trychroma.com/) (LangChain-friendly, easy to start).
  - [Weaviate](https://weaviate.io/) (scalable, REST/GraphQL APIs, hybrid search).
  - [Milvus](https://milvus.io/) (cloud-native, high-performance).
- **Managed services**:
  - [Pinecone](https://www.pinecone.io/) (fully managed vector DB, easy scaling).
  - [Qdrant Cloud](https://qdrant.tech/) (open-source + managed service).
  - [Azure Cognitive Search**, **AWS Kendra**, **Google Vertex AI Matching Engine** (cloud-native options).

## 🔹 Example (with FAISS)
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Example documents
docs = ["AI is transforming healthcare", "Transformers are powerful models"]

# Create vector store
vectorstore = FAISS.from_texts(docs, embedder)

# Query
results = vectorstore.similarity_search("How is AI used in medicine?", k=1)
print(results[0].page_content)
```


# 🌐 LangGraph

## 🚀 Overview

**LangGraph** is an open-source framework by **LangChain** for building **stateful**, **multi-agent** systems.
It lets you design agent workflows as **graphs** — where each node is an action, tool, or model, and edges define how data flows.

✅ Stateful & resumable
✅ Human-in-the-loop control
✅ Streaming & observability
✅ Fully customizable and production-ready


## ⚙️ Installation


```python
pip install -U langgraph
```

---

## 💡 Quick Start

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant."
)

response = agent.invoke({"messages": [{"role": "user", "content": "Weather in Bogotá?"}]})
print(response)
```

---

## 🧠 Core Concepts

* **Graph:** Nodes (agents/functions) + Edges (execution flow)
* **State:** Shared memory that persists between steps
* **Control:** Conditional edges, human approval, error recovery
* **Integration:** Works seamlessly with LangChain tools & LangSmith

---

## 🧩 When to Use

Use LangGraph if you need:

* Multi-agent orchestration
* Stateful, long-running logic
* Transparent, controllable agent behavior


Here’s a **professional README.md** draft that introduces **Groq**, **LCEL**, **Chain Components**, and **LangServe**, and explains how they fit together in a modern LLM-powered application:

---

# 🚀 Building Efficient LLM Applications with Groq, LCEL, and LangServe

This project demonstrates how to build **high-performance, modular, and scalable LLM applications** using:

- ⚡ **Groq** — hardware acceleration for LLM inference  
- 🧩 **LCEL (LangChain Expression Language)** — composable pipelines for building LLM workflows  
- 🔗 **Chain Components** — modular building blocks like prompts, retrievers, and memory  
- 🌐 **LangServe** — an API deployment layer for LangChain applications  

---

## 🧠 Overview

### 1. Groq — Accelerated LLM Inference
**[Groq](https://groq.com)** provides **ultra-low-latency inference** for large language models using its **Groq LPU™ (Language Processing Unit)** hardware.  
Integrating Groq into LangChain or LCEL pipelines allows:
- **Faster response times** for chatbots and RAG systems  
- **Deterministic latency** for enterprise use cases  
- Drop-in replacement for OpenAI/Anthropic API endpoints

Example:
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="mixtral-8x7b", temperature=0.2)
````

### 2. LCEL — LangChain Expression Language

**LCEL (LangChain Expression Language)** is a **declarative and composable syntax** for chaining LLM components.
It provides a functional way to define how data flows between components.

Example:

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
model = ChatGroq(model="mixtral-8x7b")

chain = {"text": RunnablePassthrough()} | prompt | model
result = chain.invoke({"text": "Hello world!"})
print(result.content)
```

Benefits:

* No need for custom Python glue code
* Easier debugging and visualization
* Works seamlessly with any LLM backend (Groq, OpenAI, etc.)

---

### 3. Chain Components — Building Blocks of LLM Workflows

LangChain uses **Chain Components** to build modular pipelines:

* **Prompt Templates** — define model input
* **LLMs / Chat Models** — the core reasoning engine
* **Retrievers / Vector Stores** — retrieve relevant context (for RAG)
* **Memory** — maintain conversation state
* **Output Parsers** — structure model responses

Example of composing components:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "Summarize the following text:\n{text}"
prompt = PromptTemplate.from_template(template)

chain = LLMChain(prompt=prompt, llm=ChatGroq(model="mixtral-8x7b"))
```

---

### 4. LangServe — Deploying LangChain Apps as APIs

**[LangServe](https://python.langchain.com/docs/langserve/)** turns any LangChain or LCEL pipeline into a production-ready REST API.
It’s built on **FastAPI** and automatically provides interactive documentation and streaming support.

Example:

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

app = FastAPI()
prompt = ChatPromptTemplate.from_template("Answer concisely: {question}")
llm = ChatGroq(model="mixtral-8x7b")

chain = prompt | llm
add_routes(app, chain, path="/qa")
```

Run locally:

```bash
uvicorn app:app --reload
```

Now your chain is accessible at:

```
GET /qa?question=What is LCEL?
```

---

## ⚙️ Putting It All Together

| Layer                 | Technology       | Purpose                                 |
| --------------------- | ---------------- | --------------------------------------- |
| **Model Inference**   | Groq             | High-speed, low-latency model execution |
| **Logic Composition** | LCEL             | Declarative chaining of components      |
| **Building Blocks**   | Chain Components | Modular architecture for flexibility    |
| **Deployment**        | LangServe        | Expose your LLM app as a production API |

---

## 🧩 Example Project Structure

```
.
├── app.py               # LangServe entrypoint
├── chains/
│   ├── translation_chain.py
│   └── rag_chain.py
├── prompts/
│   └── templates.py
├── requirements.txt
└── README.md
```

---

## 🧰 Requirements

* Python 3.10+
* `langchain`
* `langchain-groq`
* `langserve`
* `fastapi`
* `uvicorn`

Install dependencies:

```bash
pip install langchain langchain-groq langserve fastapi uvicorn
```

---

## 🧪 Quick Test

```bash
curl "http://localhost:8000/qa?question=What+is+Groq?"
```

---

## 📘 References

* [Groq API Docs](https://groq.com/)
* [LangChain Documentation](https://python.langchain.com/)
* [LangServe Guide](https://python.langchain.com/docs/langserve/)
* [LCEL Reference](https://python.langchain.com/docs/expression_language/)

---

### 💡 Summary

By combining **Groq**, **LCEL**, **Chain Components**, and **LangServe**, you can:

* Build **composable**, **performant**, and **deployable** LLM applications
* Optimize **latency** and **scalability**
* Transition smoothly from **prototype to production**


# 📘 LangChain Session History — A Practical Guide

LangChain provides multiple tools to manage **chat history**, **session memory**, and **stateful interactions**.
This guide explains how to use the key classes:

* **`ChatMessageHistory`**
* **`BaseChatMessageHistory`**
* **`RunnableWithMessageHistory`**

With examples that show how to store, retrieve, and reuse chat messages across runs.

---

# 🔧 1. `ChatMessageHistory`

`ChatMessageHistory` is the simplest in-memory message store for a chat session.

### ✔ When to use

* When you need lightweight, in-memory history
* When each session has its own message buffer
* When you want to manually add or retrieve messages

### ✔ What it stores

* `HumanMessage`
* `AIMessage`
* `SystemMessage`
* `ToolMessage`

---

## 📌 Example: Basic Usage

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = ChatMessageHistory()

# Add messages
history.add_user_message("Hello, how are you?")
history.add_ai_message("I'm good, thanks!")

# Inspect messages
for msg in history.messages:
    print(msg.type, ":", msg.content)
```

### Output:

```
human : Hello, how are you?
ai : I'm good, thanks!
```

`ChatMessageHistory` is often used inside memory wrappers or custom session managers.

---

# 🧩 2. `BaseChatMessageHistory`

This is an **abstract base class** used by LangChain to define the interface for any chat history backend.

### ✔ When to use

* When you want to build a **custom message history**, like:

  * Redis-backed history
  * DynamoDB
  * SQL
  * MongoDB
  * Local disk storage

### ✔ Must implement

```python
get_messages()
add_message()
clear()
```

---

## 📌 Example: Custom History Backend

Below is a minimal example using Python lists but showing how you would implement your own DB-backed class:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class MyCustomHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages = []

    def add_message(self, message: BaseMessage):
        self._messages.append(message)

    def get_messages(self):
        return self._messages

    def clear(self):
        self._messages = []
```

You can now plug this into any LangChain system expecting a `BaseChatMessageHistory`.

---

# 🔄 3. `RunnableWithMessageHistory`

This is the **most important modern API** for storing session history inside LangChain’s new Runnable framework.

It automatically:

* Tracks messages for each session
* Injects previous messages into the prompt
* Saves AI + user messages after each call

### ✔ When to use

* Building **stateful chatbots**
* Managing **per-session memory**
* Using LangChain’s `Runnable` pipelines
* Working with web apps (FastAPI/Flask/Streamlit)

---

## ✨ How it works

You wrap ANY runnable (e.g., an LLM) with a message history provider:

```python
RunnableWithMessageHistory(
    runnable,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

Where:

* `get_history(session_id) → ChatMessageHistory`
  returns the history for a given user
* `input_messages_key`
  tells the wrapper which key contains the user input
* `history_messages_key`
  tells the wrapper where to inject history into the prompt

---

## 📌 Complete Example: Stateful Chatbot

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# LLM
model = ChatOpenAI(model="gpt-4o-mini")

# All session histories stored here
session_store = {}

# Function required by RunnableWithMessageHistory
def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Create runnable with message history
with_history = RunnableWithMessageHistory(
    model,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Example call
result = with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user123"}},
)

print(result)
```

### Next call in the same session:

```python
with_history.invoke(
    {"input": "What did I just tell you?"},
    config={"configurable": {"session_id": "user123"}},
)
```

The model now sees both messages in context.

---

# 🧠 What Does `RunnableWithMessageHistory` Actually Inject?

Before each call, it transforms this:

```python
{"input": "..."}
```

Into:

```python
{
  "history": [
      HumanMessage("Hello!"),
      AIMessage("Hi there!")
  ],
  "input": "New question here"
}
```

And after the call, it saves:

* the user's input
* the AI's output

Into the selected history store.

---

# 🗂 Summary Table

| Component                    | Purpose                                 | When to Use                  | Persistent?             |
| ---------------------------- | --------------------------------------- | ---------------------------- | ----------------------- |
| `ChatMessageHistory`         | In-memory message buffer                | Small apps, quick prototypes | ❌ No                    |
| `BaseChatMessageHistory`     | Interface for custom history            | Custom DB backends           | ✔ Yes (if you build it) |
| `RunnableWithMessageHistory` | Wraps any runnable with session history | Stateful LLM apps            | Depends on backend      |

---

# 🏁 Final Notes

* Use **`ChatMessageHistory`** for simple apps.
* Implement **`BaseChatMessageHistory`** for production apps using databases.
* Wrap your LLM with **`RunnableWithMessageHistory`** for stateful conversation flows.
