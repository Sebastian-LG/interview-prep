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

* Python 3.8+
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
