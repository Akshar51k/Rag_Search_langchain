import os
import requests
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ['USER_AGENT'] = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/122.0.0.0 Safari/537.36'
)

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

st.set_page_config(page_title="News Research Tool", page_icon="📈")
st.title("📈 News Research Tool")


groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature = 0.5,
        api_key=groq_api_key,
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#url type detection
def is_pdf_url(url):
    if url.lower().endswith(".pdf"):
        return True
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        return "pdf" in resp.headers.get("Content-Type", "").lower()
    except Exception:
        return False

JS_HEAVY_DOMAINS = [
    "bloomberg.com", "wsj.com", "ft.com", "reuters.com",
    "nytimes.com", "washingtonpost.com", "forbes.com",
    "techcrunch.com", "theverge.com",
]

def is_dynamic_url(url):
    return any(domain in url for domain in JS_HEAVY_DOMAINS)

#url loader
def load_pdf_url(url):
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        resp = requests.get(url, timeout=30, headers={"User-Agent": os.environ["USER_AGENT"]})
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name
        docs = PyMuPDFLoader(tmp_path).load()
        for doc in docs:
            doc.metadata["source"] = url
        os.unlink(tmp_path)
        return docs, None
    except ImportError:
        return [], "PyMuPDF not installed. Run: pip install pymupdf"
    except Exception as e:
        return [], str(e)

def load_dynamic_url(url):
    try:
        from langchain_community.document_loaders import PlaywrightURLLoader
        return PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer", "nav"]).load(), None
    except ImportError:
        try:
            from langchain_community.document_loaders import SeleniumURLLoader
            return SeleniumURLLoader(urls=[url]).load(), None
        except ImportError:
            return [], "Install Playwright (pip install playwright && playwright install chromium) or Selenium (pip install selenium webdriver-manager)"
    except Exception as e:
        return [], str(e)

def load_static_url(url):
    try:
        return WebBaseLoader(
            web_paths=[url],
            header_template={"User-Agent": os.environ["USER_AGENT"]},
        ).load(), None
    except Exception as e:
        return [], str(e)

def smart_load(url):
    if is_pdf_url(url):
        docs, err = load_pdf_url(url)
        return "PDF", docs, err
    elif is_dynamic_url(url):
        docs, err = load_dynamic_url(url)
        return "Dynamic", docs, err
    else:
        docs, err = load_static_url(url)
        # fallback to dynamic if almost no content extracted
        if not err and docs and len(docs[0].page_content.strip()) < 200:
            docs, err = load_dynamic_url(url)
            return "Dynamic (fallback)", docs, err
        return "Static", docs, err



st.sidebar.title("Settings")

num_urls = st.sidebar.slider("Number of URLs", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.title("Article URLs")

urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

#init model
embeddings = load_embeddings()
llm = load_llm()

# ── Process URLs ───────────────────────────────────────────────────────────────
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        all_docs = []
        st.subheader("Loading Sources")

        for url in urls:
            with st.spinner(f"Loading: {url[:70]}..."):
                url_type, docs, error = smart_load(url)

            if error:
                st.error(f"❌ [{url_type}] {url[:60]}\n{error}")
            elif not docs:
                st.warning(f"⚠️ No content extracted from: {url[:60]}")
            else:
                all_docs.extend(docs)
                st.success(f"✅ [{url_type}] Loaded {len(docs)} page(s) from {url[:60]}")

        if not all_docs:
            st.error("Could not extract content from any URL.")
        else:
            with st.spinner("Splitting text into chunks..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap=chunk_size // 5,
                )
                docs = splitter.split_documents(all_docs)

            with st.spinner(f"Building FAISS index over {len(docs)} chunks..."):
                vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state["vectorstore"] = vectorstore

            st.success(f"Done! Indexed {len(docs)} chunks from {len(urls)} source(s).")

#prompt query and result
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

if query:
    if "vectorstore" not in st.session_state:
        st.error("Please process URLs first.")
    else:
        vectorstore = st.session_state["vectorstore"]

        with st.spinner("Searching and generating answer..."):
            retrieved_docs = vectorstore.similarity_search(query, k=3)

            template = """You are a precise research assistant. Use ONLY the context below to answer.
If the answer isn't in the context, say "I couldn't find this in the provided sources."

Context:
{summaries}

Question: {question}

Answer:"""

            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate(
                    template=template,
                    input_variables=["summaries", "question"]
                )},
                return_source_documents=True,
            )

            result = chain.invoke({"question": query})

        st.subheader("Answer")
        st.write(result.get("answer", "No answer returned."))

        # Sources extracted from retrieved doc metadata (reliable)
        sources = list({
            doc.metadata.get("source", "")
            for doc in retrieved_docs
            if doc.metadata.get("source", "").strip()
        })

        if sources:
            st.subheader("Sources")
            for source in sources:
                st.write(source)

        with st.expander("View retrieved context chunks"):
            for i, doc in enumerate(retrieved_docs, 1):
                st.markdown(f"**Chunk {i}** — `{doc.metadata.get('source', 'unknown')}`")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                st.divider()
