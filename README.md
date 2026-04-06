# RAG Email Assistant

A privacy-preserving email draft assistant powered by local LLMs and Retrieval-Augmented Generation (RAG). No data leaves your machine — fully FERPA/HIPAA-friendly.

---

## How it works

```
Incoming email
      ↓
Semantic search across two knowledge bases
      ↓
┌─────────────────┐     ┌─────────────────┐
│  Past email     │     │  Website data   │
│  archive        │     │  (scraped)      │
└────────┬────────┘     └────────┬────────┘
         └──────────┬────────────┘
                    ↓
         Llama 3 (via Ollama) drafts reply
                    ↓
         Review and send
```

Past emails teach the system **tone and format**. Website data provides **accurate, up-to-date facts**. The LLM combines both to generate a contextually relevant draft.

---

## Tech stack

| Component | Tool |
|---|---|
| LLM | Llama 3 via Ollama (runs locally) |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Orchestration | LangChain |
| Web scraping | BeautifulSoup |
| UI | Streamlit |
| Knowledge base | Word document (.docx) + scraped URLs |

---

## Features

- **Fully local** — Llama 3 runs on your machine via Ollama. No API keys, no cloud, no data exposure
- **Dual knowledge base** — retrieves from past email pairs AND live website content simultaneously
- **One-click refresh** — re-ingest emails or re-scrape websites from the UI sidebar
- **Source transparency** — expandable "Sources used" section shows exactly what context was retrieved
- **Privacy by design** — suitable for institutional use with sensitive applicant data

---

## Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running
- Llama 3 pulled: `ollama pull llama3`

### Install

```bash
git clone https://github.com/atharva9520-hub/rag_email_assistant.git
cd rag_email_assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare your knowledge base

**Email pairs** — create `emails/knowledge_base.docx` with this format:

```
Question:
[paste incoming email here]

response:
[paste your reply here]

Question:
[next email...]
```

**Website URLs** — add one URL per line to `urls.txt`:

```
https://yourorganization.com/admissions
https://yourorganization.com/financial-aid
```

### Ingest data

```bash
# Load email pairs into ChromaDB
python app/ingest.py

# Scrape website URLs into ChromaDB
python app/scraper.py
```

### Run

```bash
streamlit run app/main.py --server.fileWatcherType none
```

Open `http://localhost:8501` in your browser.

---

## Project structure

```
rag_email_assistant/
├── app/
│   ├── main.py          # Streamlit UI
│   ├── ingest.py        # Email pair ingestion into ChromaDB
│   └── scraper.py       # Website scraper into ChromaDB
├── emails/              # Your email knowledge base (gitignored)
├── data/                # ChromaDB vector store (gitignored)
├── urls.txt             # URLs to scrape
├── requirements.txt
└── .gitignore
```

---

## Privacy

- All processing happens locally on your machine
- Email data is stored only in the local ChromaDB instance
- No data is sent to external APIs or cloud services
- The `emails/` and `data/` directories are gitignored by default

---

## Future improvements

- [ ] Folder watch automation — auto-process emails dropped into a folder
- [ ] Word document output — save drafts as `.docx` files
- [ ] Outlook integration — connect via Microsoft Graph API (requires admin consent)
- [ ] Confidence scoring — flag low-confidence replies for closer review
- [ ] Multi-language support

---

## License

MIT
