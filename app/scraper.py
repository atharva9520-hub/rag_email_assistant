import requests
import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
URLS_FILE    = "urls.txt"
CHROMA_PATH  = "data/chroma"
COLLECTION_WEB = "website_content"

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to ChromaDB ───────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=CHROMA_PATH)

def scrape_url(url):
    """Scrape text content from a single URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove nav, footer, scripts, styles — keep main content only
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()
        
        # Get clean text
        text = soup.get_text(separator="\n", strip=True)
        
        # Remove excessive blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)
        
        return clean_text
    
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def chunk_text(text, url, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append({
                "text": chunk,
                "source": url
            })
    
    return chunks

def ingest_website():
    """Scrape all URLs and store in ChromaDB."""
    
    # Read URLs
    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(urls)} URLs to scrape")
    
    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection(COLLECTION_WEB)
        print("Cleared existing website collection")
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_WEB,
        metadata={"hnsw:space": "cosine"}
    )
    
    all_chunks = []
    
    for url in urls:
        print(f"Scraping: {url}")
        text = scrape_url(url)
        
        if text:
            chunks = chunk_text(text, url)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks extracted")
        else:
            print(f"  → Skipped")
    
    if not all_chunks:
        print("No content scraped — check your urls.txt")
        return
    
    # Embed and store all chunks
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    
    documents  = []
    metadatas  = []
    ids        = []
    embeddings = []
    
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.encode(chunk["text"]).tolist()
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "type": "website"})
        ids.append(f"web_{i}")
        embeddings.append(embedding)
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    
    print(f"Successfully ingested {len(all_chunks)} website chunks into ChromaDB")
    print(f"Stored at: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_website()