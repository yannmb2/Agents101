# 📌 Project 1: Agentic RAG with Verification

🎯 **Objective**

Build an intelligent agent that receives a user question, retrieves relevant information from a document base (using RAG), answers the question, and then verifies whether the answer is truly supported by the retrieved context.

## 🏗️ Architecture

Your agent follows a three-node graph structure:

### 🔸 Retrieval Node
- Uses FAISS + Mistral embeddings to find the most relevant documents based on the user's question
- **Output**: the question and the retrieved context

### 🔸 Answer Node  
- Passes the question and context to Mistral LLM to generate a helpful answer
- **Output**: the original question, context, and generated answer

### 🔸 Verifier Node
- Uses Mistral LLM with a specific system prompt: *"Is the following answer well-supported by the provided context? Answer yes or no, and justify briefly."*
- Based on the result, either confirms the answer ✅ or flags it as unreliable ⚠️

## 🚀 Setup

### 1. Install Dependencies
```bash
poetry install
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Prepare Documents
Create a `Data/` folder and add your documents:
```bash
mkdir Data
# Add your .txt and .pdf files to the Data/ folder
```

## 🎮 Usage

### Command Line Interface

#### Standard RAG (Console)
```bash
poetry run python main.py
```

#### Direct RAG Module
```bash
poetry run python mistral_rag.py
```

### Web Interface (Streamlit)

Launch the interactive web application:
```bash
poetry run streamlit run streamlit_app.py
```

The Streamlit app provides:
- **Dual Mode**: Simple RAG vs. RAG with Verification
- **Interactive Chat**: Web-based conversation interface
- **Context Transparency**: View retrieved documents for each answer
- **Verification Analysis**: See reliability assessment for each response
- **Document Management**: Overview of loaded documents in sidebar

## 📁 Project Structure

```
agents101/
├── pyproject.toml          # Dependencies configuration
├── main.py                 # Main entry point (console)
├── mistral_rag.py         # Core RAG implementation
├── streamlit_app.py       # Web interface
├── .env                   # API keys (create this)
├── Data/                  # Document storage
│   ├── document1.pdf
│   ├── document2.txt
│   └── ...
└── README.md              # This file
```

## 🔧 Features

- **Multi-format Support**: PDF and TXT documents
- **Persistent Vector Store**: FAISS index with automatic save/load
- **Conversation History**: Maintains context across multiple questions
- **Verification System**: Automatic answer reliability checking
- **Batch Processing**: Efficient document embedding with progress tracking
- **Error Handling**: Graceful handling of API errors and missing files

## 🛠️ Technical Stack

- **LLM**: Mistral AI (mistral-large-latest)
- **Embeddings**: Mistral