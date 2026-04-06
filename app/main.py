import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH        = "data/chroma"
COLLECTION_EMAIL   = "email_pairs"
COLLECTION_WEB     = "website_content"
MODEL_NAME         = "llama3"
TOP_K              = 3

# ── Load models (cached so they don't reload on every interaction) ────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return OllamaLLM(model=MODEL_NAME)

@st.cache_resource
def load_collections():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    email_col = client.get_collection(COLLECTION_EMAIL)
    web_col   = client.get_collection(COLLECTION_WEB)
    return email_col, web_col

# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_context(query, email_col, web_col, embedder):
    query_embedding = embedder.encode(query).tolist()
    
    # Search email archive
    email_results = email_col.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )
    
    # Search website content
    web_results = web_col.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )
    
    # Build context string
    context_parts = []
    
    context_parts.append("=== PAST EMAIL RESPONSES (use for tone and format) ===")
    for i, (doc, meta) in enumerate(zip(
        email_results["documents"][0],
        email_results["metadatas"][0]
    )):
        context_parts.append(f"\nExample {i+1}:")
        context_parts.append(f"Question: {doc}")
        context_parts.append(f"Answer: {meta['answer']}")
    
    context_parts.append("\n=== WEBSITE INFORMATION (use for accurate facts) ===")
    for i, (doc, meta) in enumerate(zip(
        web_results["documents"][0],
        web_results["metadatas"][0]
    )):
        context_parts.append(f"\n[Source: {meta['source']}]")
        context_parts.append(doc)
    
    return "\n".join(context_parts)

# ── Draft reply ───────────────────────────────────────────────────────────────
def draft_reply(incoming_email, context, llm):
    prompt = f"""You are an admissions assistant at the La Follette School of Public Affairs, University of Wisconsin-Madison.

Your job is to draft a professional, warm, and helpful reply to the incoming email below.

Use the past email examples to match the tone and format of previous responses.
Use the website information to ensure your reply contains accurate, up-to-date facts.

Do NOT make up information. If you are unsure about something, say so politely and suggest the applicant visit the website or contact the office directly.

{context}

=== INCOMING EMAIL TO REPLY TO ===
{incoming_email}

=== DRAFT REPLY ===
"""
    return llm.invoke(prompt)

# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="La Follette Email Assistant",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 La Follette Email Assistant")
    st.caption("Privacy-preserving RAG — all data stays local, nothing sent to the cloud")
    
    # Load resources
    with st.spinner("Loading models..."):
        embedder   = load_embedder()
        llm        = load_llm()
        email_col, web_col = load_collections()
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Incoming Email")
        incoming = st.text_area(
            "Paste the email you received here",
            height=400,
            placeholder="Dear Admissions Team,\n\nI am interested in applying to the MPA program..."
        )
        
        generate = st.button("Generate Draft Reply", type="primary")
    
    with col2:
        st.subheader("Draft Reply")
        
        if generate:
            if not incoming.strip():
                st.warning("Please paste an incoming email first.")
            else:
                with st.spinner("Retrieving context and generating reply..."):
                    context = retrieve_context(
                        incoming, email_col, web_col, embedder
                    )
                    reply = draft_reply(incoming, context, llm)
                
                st.text_area(
                    "Generated draft — review before sending",
                    value=reply,
                    height=400
                )
                
                # Show sources used
                with st.expander("Sources used for this reply"):
                    st.text(context)
        else:
            st.info("Paste an email and click Generate to get a draft reply.")
    
    # Sidebar
    with st.sidebar:
        st.header("Knowledge Base")
        st.metric("Email pairs", email_col.count())
        st.metric("Website chunks", web_col.count())
        
        st.divider()
        st.header("Refresh Data")
        
        if st.button("Re-ingest emails"):
            import subprocess
            subprocess.run(["python", "app/ingest.py"])
            st.success("Emails re-ingested!")
            st.cache_resource.clear()
        
        if st.button("Re-scrape websites"):
            import subprocess
            subprocess.run(["python", "app/scraper.py"])
            st.success("Websites re-scraped!")
            st.cache_resource.clear()
        
        st.divider()
        st.caption("All data processed locally. No information sent to external servers.")

if __name__ == "__main__":
    main()