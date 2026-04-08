# AgentMail — Agentic RAG Email Assistant

A privacy-preserving, multi-model agentic email draft assistant powered by local LLMs and Retrieval-Augmented Generation. Two language models reason against each other to generate accurate, hallucination-free email drafts. No data leaves your machine — fully FERPA-compliant.

---

## Architecture

```
Incoming email
      ↓
Node 1: Classifier (Mistral)
  - Extracts topic and keywords dynamically from email content
  - Decides if web search is needed
      ↓
Node 2: Retriever (ChromaDB)
  - Past email pairs → tone and format reference only
  - Scraped website content → factual grounding only
      ↓
Node 3: Web Search (DuckDuckGo)
  - Site-restricted to official domain only
  - Triggered only when local retrieval is insufficient
      ↓
Node 4: Drafter (Llama 3)
  - Generates reply using tone from emails, facts from website
      ↓
Node 5: Critic (Mistral)
  - Independently evaluates accuracy, completeness, tone,
    clarity, and hallucinations
  - Returns specific critique to Llama 3 (max 2 loops)
      ↓
Node 6: Judge (Mistral)
  - Makes final approval decision
  - Flags low confidence replies for human review
      ↓
Final draft with quality score
```

---

## Why two models?

A single model evaluating its own output has self-approval bias — it tends to approve what it wrote. By using Mistral as an independent critic and judge, the system catches hallucinations and gaps that the drafter missed.

| Model | Role | Strength |
|---|---|---|
| Llama 3 | Drafter and reviser | Natural, warm prose generation |
| Mistral | Classifier, critic, judge | Analytical reasoning and evaluation |

---

## Dual knowledge base design

Two separate ChromaDB collections with deliberately different purposes:

| Collection | Source | Purpose |
|---|---|---|
| Email archive | Word document (.docx) | Tone, greeting, sign-off format only |
| Website content | BeautifulSoup scraper | All factual claims — courses, deadlines, requirements |

This separation prevents outdated email information from being used as facts and forces all factual claims to be grounded in current official website content.

---

## Tech stack

| Component | Tool |
|---|---|
| LLM — drafting | Llama 3 via Ollama |
| LLM — reasoning | Mistral via Ollama |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Orchestration | LangChain |
| Web scraping | BeautifulSoup |
| Web search | DuckDuckGo (site-restricted) |
| UI | Streamlit (basic RAG version) |
| Knowledge base input | Word document (.docx) |

---

## Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running
- Models pulled locally:
```bash
ollama pull llama3
ollama pull mistral
```

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
https://yourorganization.com/programs
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

**Agentic pipeline (terminal):**
```bash
python app/agent.py
```

**Basic RAG UI (Streamlit):**
```bash
streamlit run app/main.py --server.fileWatcherType none
```

---

## Project structure

```
rag_email_assistant/
├── app/
│   ├── agent.py         Multi-model agentic pipeline
│   ├── main.py          Streamlit UI (basic RAG version)
│   ├── ingest.py        Email pair ingestion into ChromaDB
│   └── scraper.py       Website scraper into ChromaDB
├── emails/              Email knowledge base (gitignored)
├── data/                ChromaDB vector store (gitignored)
├── urls.txt             URLs to scrape
├── requirements.txt
└── .gitignore
```

---

## Privacy

- Llama 3 and Mistral run entirely on-device via Ollama
- All embeddings generated locally by sentence-transformers
- ChromaDB stores all vectors as local files
- Email data never transmitted to any external service
- Web search queries only the official public website
- `emails/` and `data/` directories are gitignored by default
- Suitable for FERPA-protected student data

---

## Sample output

```
MULTI-MODEL AGENTIC RAG EMAIL ASSISTANT
Drafter: llama3 | Critic: mistral

[Node 1] Mistral classifying email...
  Topic:        program prerequisites
  Keywords:     MIPA prerequisites, core courses, requirements
  Needs search: yes

[Node 2] Retrieving from knowledge base...
  Email matches: 3
  Web matches:   3

[Node 3] Searching official domain...
  Web search: 3 results

[Node 4] Llama 3 drafting reply (loop 1)...

[Node 5] Mistral critiquing draft (loop 1)...
  Accuracy:       5/5
  Completeness:   4/5
  Tone:           5/5
  Clarity:        5/5
  Hallucinations: None detected
  Overall:        4.75/5
  Approved:       Yes

[Node 6] Mistral making final judgement...
  Final score:    5/5
  Summary:        Ready for human review

FINAL DRAFT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[complete email draft with bullet points]

Overall quality:  5/5
STATUS: Ready for human review
```

---

## Planned improvements

- [ ] Connect agent to Streamlit UI
- [ ] Word document output — save drafts as `.docx` files
- [ ] Folder watch automation — auto-process emails dropped into a watched folder
- [ ] Outlook integration via Microsoft Graph API
- [ ] Auto-extract hyperlinks from email pairs and add to scraper
- [ ] Confidence-based routing — low confidence emails flagged for senior review

---

## License

MIT
