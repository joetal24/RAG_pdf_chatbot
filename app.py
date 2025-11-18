# app.py
import os
import streamlit as st
from rag_utils import extract_text_from_pdf, chunk_text, load_hf_embedding_model, get_chroma_client, create_chroma_collection, embed_and_store_chunks, retrieve_similar
import chromadb

st.set_page_config(page_title="RAG PDF QA (simple)", layout="wide")

st.title("RAG PDF Q&A — Simple (LangChain-style)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save temp
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf("uploaded.pdf")
    st.write("Preview (first 1000 chars):")
    st.code(text[:1000])

    # Chunk text
    st.info("Splitting text into chunks...")
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    st.write(f"Created {len(chunks)} chunks.")

    # Load embedding model
    model = load_hf_embedding_model()
    st.success("Embedding model loaded.")

    # Setup Chroma
    client, persist_dir = get_chroma_client()
    # Note: chroma client settings can be configured; here we use default in-memory (for persistent use, pass proper settings)
    try:
        collection = client.get_collection(name="pdf_rag")
    except Exception:
        collection = client.create_collection(name="pdf_rag")

    st.info("Embedding chunks and storing in vector DB (Chroma)...")
    embed_and_store_chunks(chunks, model, collection)
    st.success("Chunks embedded and stored.")

    # Ask question
    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        st.info("Searching for relevant chunks...")
        retrieved = retrieve_similar(collection, user_question, model, top_k=4)
        st.write("Top relevant snippets:")
        for i, r in enumerate(retrieved):
            st.markdown(f"**Snippet {i+1}** (meta: {r['meta']}):")
            st.write(r["text"][:800])  # show up to 800 chars

        # Build prompt for LLM (Grok)
        context_text = "\n\n---\n\n".join([r["text"] for r in retrieved])
        prompt = f"""You are an assistant. Use the following document excerpts to answer the question accurately.

        CONTEXT:
        {context_text}

        QUESTION:
        {user_question}

        ANSWER (use only the information from CONTEXT; if not present, say you don't know):
        """

        st.write("Sending to Grok LLM...")
        # --- Simple Grok call via requests (example) ---
        grok_key = os.environ.get("GROK_API_KEY")
        if not grok_key:
            st.warning("GROK_API_KEY not set. Set it in environment to query Grok. Showing assembled prompt instead.")
            st.code(prompt[:2000])
        else:
            import requests
            grok_url = "https://api.grok.api/llm"  # <-- placeholder, replace with real Grok endpoint/SDK
            headers = {"Authorization": f"Bearer {grok_key}", "Content-Type": "application/json"}
            payload = {"prompt": prompt, "max_tokens": 400}
            # NOTE: adapt payload to the real Grok API shape
            resp = requests.post(grok_url, headers=headers, json=payload)
            if resp.ok:
                ans = resp.json().get("text") or resp.json().get("answer") or resp.text
                st.success("Grok answer:")
                st.write(ans)
            else:
                st.error(f"Grok API error: {resp.status_code} - {resp.text}")

