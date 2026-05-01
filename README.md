# 🏥 Medical Chatbot — AI-Powered Health Assistant


<p align="center">
  A production-ready, Retrieval-Augmented Generation (RAG) medical chatbot that answers health-related questions using knowledge extracted from medical PDF documents, powered by LLaMA 3.1 via Groq and stored in a Pinecone vector database.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Indexing Your Medical Data](#indexing-your-medical-data)
  - [Running the App](#running-the-app)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [License](#-license)

---

## 🧠 Overview

The **Medical Chatbot** is an end-to-end Generative AI application designed to provide concise, context-aware answers to medical questions. It ingests medical PDF documents, converts them into semantic vector embeddings, stores them in Pinecone, and then uses a RAG pipeline to retrieve relevant context and generate accurate answers using the **LLaMA 3.1 8B Instant** model served through **Groq**.

The chatbot is accessible through a clean, modern web interface styled as a "Dr. AI Assistant" chat window.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Flask Web App (app.py)
    │
    ▼
LangChain RAG Pipeline
    ├── Retriever ──────► Pinecone Vector Store
    │                         (384-dim cosine index)
    │                              ▲
    │                    HuggingFace Embeddings
    │                    (all-MiniLM-L6-v2)
    │
    └── LLM ────────────► Groq API
                          (llama-3.1-8b-instant)
                               │
                               ▼
                         Final Answer → User
```

**Data Ingestion Flow (one-time setup via `store_index.py`):**

```
PDF Files (./Data/)
    │
    ▼
PyPDFLoader (LangChain)
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=500, overlap=20)
    │
    ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2 → 384 dimensions)
    │
    ▼
Pinecone Index ("medicalbot")
```

---

## ✨ Features

- 🔍 **Semantic Search** — Uses cosine similarity over 384-dimensional embeddings to find the most relevant medical content for each query
- 🤖 **LLM-Powered Answers** — Leverages LLaMA 3.1 8B Instant via Groq for fast, intelligent responses
- 📄 **PDF Knowledge Base** — Easily extendable by dropping new PDF documents into the `Data/` folder and re-indexing
- 💬 **Real-time Chat UI** — Responsive chat interface with typing indicators, auto-scroll, and smooth UX
- 🔒 **Honest AI** — Instructs the model to say "I don't know" rather than hallucinate answers
- ⚡ **Fast Inference** — Groq's hardware acceleration enables near-instant LLM responses
- 🌐 **REST API** — Simple `/get` endpoint for easy integration with other frontends

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Flask 3.0.3 |
| **LLM** | LLaMA 3.1 8B Instant (via Groq) |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace, 384-dim) |
| **Vector Database** | Pinecone (Serverless, AWS us-east-1) |
| **RAG Framework** | LangChain 0.2.16 |
| **PDF Loader** | PyPDF + LangChain DirectoryLoader |
| **Text Splitter** | RecursiveCharacterTextSplitter |
| **Frontend** | HTML5, CSS3 (custom), Vanilla JS |
| **Icons & Fonts** | FontAwesome 6.4, Google Fonts (Inter) |

---

## 📁 Project Structure

```
Medical-Chatbot/
│
├── app.py                   # Main Flask application & RAG chain setup
├── store_index.py           # One-time script: ingests PDFs → Pinecone
├── summary.py               # summary/documentation for this project
├── setup.py                 # Package setup
├── requirements.txt         # Python dependencies
├── .env                     # API keys
│
├── src/
│   ├── __init__.py
│   ├── helper.py            # Core functions: load_pdf, text_split, embeddings
│   └── prompt.py            # System prompt for the LLM
│
├── templates/
│   └── chat.html            # Chat UI HTML
│
├── static/
│   └── style.css            # Custom CSS styling
│
├── Data/                    # Place your medical PDF files here
│
└── research/
    └── trials.ipynb         # Experimentation & prototyping notebook
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- A [Pinecone](https://www.pinecone.io/) account (free tier works)
- A [Groq](https://console.groq.com/) API key (free tier available)

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/Medical-Chatbot.git
cd Medical-Chatbot
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install all dependencies:**
```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (or rename the existing one) with your actual API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Security Warning:** Never commit your `.env` file to version control. It is already listed in `.gitignore`.

### Indexing Your Medical Data

**1. Add your medical PDF files to the `Data/` directory:**
```bash
# Example:
cp my_medical_textbook.pdf Data/
```

**2. Run the indexing script to embed and store them in Pinecone:**
```bash
python store_index.py
```

This will:
- Load all PDFs from the `Data/` folder
- Split them into 500-token chunks with 20-token overlap
- Generate embeddings using `all-MiniLM-L6-v2`
- Create a `medicalbot` index in Pinecone (if it doesn't exist)
- Upload all vectors to Pinecone

> ℹ️ You only need to run `store_index.py` once, or whenever you add new documents to the knowledge base.

### Running the App

```bash
python app.py
```

The server will start on **`http://0.0.0.0:8080`**. Open your browser and navigate to:

```
http://localhost:8080
```

---

## ⚙️ How It Works

1. **User sends a message** via the chat interface
2. **Flask** receives the request at the `/get` endpoint
3. **LangChain retriever** queries Pinecone using semantic similarity, fetching the top **3 most relevant document chunks**
4. **The retrieved context + user question** are assembled into a prompt using `ChatPromptTemplate`
5. **Groq's LLaMA 3.1** generates a concise answer (max 3 sentences) grounded in the retrieved context
6. **The answer is returned** to the frontend and displayed in the chat window

---

## 🔧 Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `model` | `app.py` | `llama-3.1-8b-instant` | Groq LLM model |
| `temperature` | `app.py` | `0.4` | LLM creativity (0 = deterministic) |
| `search_kwargs["k"]` | `app.py` | `3` | Number of retrieved document chunks |
| `chunk_size` | `src/helper.py` | `500` | Token size of each text chunk |
| `chunk_overlap` | `src/helper.py` | `20` | Overlap between consecutive chunks |
| `embedding model` | `src/helper.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `index_name` | `app.py` / `store_index.py` | `medicalbot` | Pinecone index name |
| `port` | `app.py` | `8080` | Flask server port |

To customize the chatbot's behavior or persona, edit the system prompt in `src/prompt.py`.

---

## 📜 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.

---

<p align="center">
  Built with ❤️ by <strong>Ramy Ahmed</strong>
</p>