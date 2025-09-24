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
