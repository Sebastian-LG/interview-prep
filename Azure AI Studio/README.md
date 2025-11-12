# 🧠 Azure AI Studio (Azure AI Foundry) Overview

## 🔷 What It Is

Azure AI Studio — now **Azure AI Foundry** — is a unified platform combining tools such as:

* **Azure Machine Learning**
* **Azure OpenAI Service**
* **Azure AI Language & Vision**

It enables you to **build, test, and deploy** AI solutions using prebuilt and custom models.

---

## 🧩 Requirements

Before starting, you need:

* An **Azure subscription**
* **Azure Storage**
* **AI resources** (e.g., Azure OpenAI resource, Azure AI Search, etc.)

---

## 🚀 Core Capabilities

* Model catalog with **benchmarks and metrics**
* **Deploy & test** models directly in Azure AI
* Integration with **Content Safety** tools

---

## ⚙️ Typical Workflow

### 1. Create a New Project

Set up a workspace and deploy one of the available **base models**.
 
### 2. Use the Playground

Experiment interactively with:

* **System message (context)**
* **Context variables**
* **Language / Subscription configuration**

**Model parameters:**

| Parameter           | Description                       |
| ------------------- | --------------------------------- |
| `max_response`      | Limits the output length          |
| `temperature`       | Controls randomness               |
| `top_p`             | Sampling parameter                |
| `stop_sequence`     | Stops generation at defined token |
| `frequency_penalty` | Reduces repetition of tokens      |
| `presence_penalty`  | Encourages new topic introduction |

You can also configure and test a **chatbot** setup here.

---

## 🔄 Prompt Flow

Create or load a **prompt flow** consisting of multiple steps, such as:

1. Extracting the query
2. Running **Python** scripts or preprocessing logic
3. Sending refined input to the chat model

---

## 📂 Add Your Data

Integrate your own data sources with:

* **Search type:** Semantic or Keyword
* Option for **strict adherence** to document content

---

## 🌐 Deployment Options

Once your solution is ready, you can deploy via:

* **Web App** (interactive chatbot UI)
* **Playground** (testing environment)
* **API** endpoint — includes code samples to query the model programmatically

---

## ✨ Enhancements

You can extend your model with:

* Additional data connections
* Prompt optimization
* Custom logic or Python components



AI Search for vectorization