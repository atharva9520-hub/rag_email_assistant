import os
import docx2txt
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# ── Config ────────────────────────────────────────────────────────────────────
EMAILS_PATH = "emails/lafollette_emails.docx"
CHROMA_PATH = "data/chroma"
COLLECTION_EMAIL = "email_pairs"

# ── Load embedding model (runs locally, no data leaves machine) ───────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to ChromaDB ───────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=CHROMA_PATH)

def parse_email_doc(path):
    """Parse Word doc into list of {question, answer} dicts."""
    raw = docx2txt.process(path)
    pairs = []
    
    # Normalize whitespace
    raw = raw.replace('\xa0', ' ')
    
    # Split into blocks — try multiple possible separators
    import re
    blocks = re.split(r'\n\s*\n(?=Question:|question:)', raw)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Case-insensitive match for question/answer labels
        # Support both "answer:" and "response:"
        q_match = re.search(r'(?i)question:\s*', block)
        a_match = re.search(r'(?i)(answer:|response:)\s*', block)
        
        if q_match and a_match:
            try:
                q_start = q_match.end()
                a_label_start = a_match.start()
                a_start = a_match.end()
                
                question = block[q_start:a_label_start].strip()
                answer = block[a_start:].strip()
                
                if question and answer:
                    pairs.append({
                        "question": question,
                        "answer": answer
                    })
            except Exception as e:
                print(f"Skipping block due to error: {e}")
                continue
    
    return pairs

def ingest_emails():
    """Load email pairs into ChromaDB."""
    
    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection(COLLECTION_EMAIL)
        print("Cleared existing email collection")
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_EMAIL,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Parsing {EMAILS_PATH}...")
    pairs = parse_email_doc(EMAILS_PATH)
    print(f"Found {len(pairs)} email pairs")
    
    if not pairs:
        print("No pairs found — check your Word doc format")
        return
    
    # Embed questions — we search on questions, retrieve answers
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for i, pair in enumerate(pairs):
        # Embed the question for search
        embedding = embedder.encode(pair["question"]).tolist()
        
        documents.append(pair["question"])
        metadatas.append({"answer": pair["answer"], "source": "email_archive"})
        ids.append(f"email_{i}")
        embeddings.append(embedding)
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    
    print(f"Successfully ingested {len(pairs)} email pairs into ChromaDB")
    print(f"Stored at: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_emails()