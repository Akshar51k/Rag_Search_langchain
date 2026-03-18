import os
import streamlit as st
import time

from dotenv import load_dotenv
load_dotenv()

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# UI
st.title("ChatBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Groq LLM (reads GROQ_API_KEY from .env file)
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    api_key=groq_api_key,
    max_tokens=1000,
)

# Embeddings (HuggingFace — works on Streamlit Cloud)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_community.document_loaders import WebBaseLoader

# Process URLs
if process_url_clicked and urls:
    loader = WebBaseLoader(
        web_paths=urls,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
    )

    data = loader.load()
    main_placeholder.text("Data Loading...Started...✅")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    main_placeholder.text("Text Splitting...Started...✅")
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("Error: Could not extract any text from the URLs. The websites might be blocking access.")
    else:
        # Build FAISS index
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding + FAISS Index Created...✅")
        time.sleep(1)
    
        # Save to session state (no disk needed)
        st.session_state["vectorstore"] = vectorstore
        st.success("Processing Complete!")

# Query input
query = st.text_input("Ask a Question:")

if query:
    if "vectorstore" in st.session_state:
        vectorstore = st.session_state["vectorstore"]

        template = """You are a helpful assistant. Use the following pieces of context to answer the user's question.
If the context contains the answer, extract it. If it doesn't, just say you don't know.
ALWAYS include the source URL exactly as provided.

Context: {summaries}

Question: {question}

Answer with Sources:"""
        PROMPT = PromptTemplate(
            template=template, input_variables=["summaries", "question"]
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
        )

        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        # Sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
    else:
        st.error("Please process URLs first!")