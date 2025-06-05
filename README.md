# 📌 Project 1: Agentic RAG with Verification

🎯 Objective
Build an intelligent agent that receives a user question, retrieves relevant information from a document base (using RAG), answers the question, and then verifies whether the answer is truly supported by the retrieved context.

🔹 Your agent could follow a three-node graph structure:

🔸 Retrieval Node
Use a retriever (e.g., FAISS + OpenAI embeddings) to find the most relevant documents based on the user's question.
Output: the question and the retrieved context.

🔸 Answer Node
Pass the question and context to an LLM (e.g., GPT-4o) to generate a helpful answer.
Output: the original question, context, and generated answer.

🔸 Verifier Node
Use another LLM (can be the same model) with a specific system prompt:
“Is the following answer well-supported by the provided context? Answer yes or no, and justify briefly.”
Based on the result, either confirm the answer or flag it as unreliable.

## How to set it up ?
```bash
poetry install
```

## How to run it ? 
```bash
poetry run python .\main.py
```

## How to create vector database and retrieve ? 
```bash
poetry run python .\mistral_rag.py
```

## ⚠️ Il faut configurer un fichier .env
```texte
MISTRAL_API_KEY = put_your_api_key_here

```