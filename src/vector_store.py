import os
import pickle
from  src.config import settings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def build_vector_store(docs: list[str], persist_dir: str = "data/faiss_index"):
    """Generate FAISS vector store from raw documents and save to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    documents = splitter.create_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=settings['OPENAI_API_KEY'], model=settings['MODEL_DEFAULT_EMBEDDING'])

    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create directory if not exists
    os.makedirs(persist_dir, exist_ok=True)
    faiss_file = os.path.join(persist_dir, "index.faiss")
    pkl_file = os.path.join(persist_dir, "index.pkl")

    vectorstore.save_local(persist_dir)  # saves index.faiss and index.pkl
    print(f"FAISS index saved to {persist_dir}")
    return vectorstore

def load_vector_store(persist_dir: str = "data/faiss_index"):
    """Load FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
