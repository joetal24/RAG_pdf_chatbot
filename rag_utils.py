# rag_utils.py
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

# --- PDF text extraction ---
def extract_text_from_pdf(path: str) -> str:
    """Return all text in the PDF as one string."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
    return "\n\n".join(texts)

# --- Simple chunking ---
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    chunk_size: approx number of words per chunk (not tokens) for simplicity.
    overlap: number of words to overlap to keep context between chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

# --- Embedding model loader (Hugging Face SentenceTransformer) ---
def load_hf_embedding_model(model_name: str = None):
    model_name = model_name or os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {model_name} (this may take a moment)")
    return SentenceTransformer(model_name)

# --- Create or connect to Chroma vector DB (local) ---
def get_chroma_client(persist_directory: str = None):
    persist_directory = persist_directory or os.environ.get("CHROMA_DB_DIR", "./chroma_db")
    client = chromadb.Client()
    return client, persist_directory

def create_chroma_collection(client, collection_name: str = "pdf_rag", persist_directory: str = None, embedding_fn=None):
    # If embedding_fn is provided, pass it. For simplicity, we'll store the embeddings ourselves.
    # Use a named collection for our docs.
    return client.create_collection(name=collection_name)

# --- Embed text chunks and add to Chroma ---
def embed_and_store_chunks(chunks: List[str], model, collection, namespace: str = None):
    """
    model: SentenceTransformer instance
    collection: chroma collection object
    """
    ids = []
    metadatas = []
    embeddings = []
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        metadatas.append({"chunk_index": i, "text_snippet": chunk[:200]})
        emb = model.encode(chunk).tolist()
        embeddings.append(emb)
    # upsert into collection
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks, )
    return True

# --- Simple retrieval function using Chroma kNN ---
def retrieve_similar(collection, query: str, model, top_k: int = 4) -> List[Dict]:
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    # results is a dict: 'ids','distances','metadatas','documents'
    docs = []
    for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
        docs.append({"text": doc, "meta": meta})
    return docs
