# News Research Tool 📈

A powerful, user-friendly web application built with Streamlit and LangChain that allows users to input news article URLs, process their content, and ask questions to get precise, source-backed answers. The tool uses the lightning-fast **Groq API** (Llama 3.3 70B model) in the backend and **HuggingFace** embeddings for semantic search.

## ✨ Features

- **Process Multiple URLs:** Input up to 3 news article URLs simultaneously.
- **Smart Data Extraction:** Automatically extracts and splits text from web pages using LangChain's `UnstructuredURLLoader`.
- **Fast Similarity Search:** Uses `FAISS` vector store in memory for rapid retrieval of relevant information.
- **High-Performance LLM:** Powered by Groq's `llama-3.3-70b-versatile` model for highly accurate, contextual answers.
- **Source Citations:** Every answer comes with the exact source URL(s) it was derived from.
- **Cloud Ready:** Designed to be easily deployed on Streamlit Community Cloud.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM/Orchestration:** [LangChain](https://www.langchain.com/)
- **API Provider:** [Groq](https://groq.com/)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS

## 🚀 Installation and Setup (Local)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
