import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from duckduckgo_search import DDGS

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH      = "data/chroma"
COLLECTION_EMAIL = "email_pairs"
COLLECTION_WEB   = "website_content"
DRAFTER_MODEL    = "llama3"
CRITIC_MODEL     = "mistral"
SITE_RESTRICT    = "site:lafollette.wisc.edu"
MAX_LOOPS        = 2
TOP_K            = 3

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
drafter  = OllamaLLM(model=DRAFTER_MODEL)   # Llama 3 — writes the reply
critic   = OllamaLLM(model=CRITIC_MODEL)    # Mistral — critiques and judges
print(f"Drafter: {DRAFTER_MODEL}")
print(f"Critic:  {CRITIC_MODEL}")

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    email_col = client.get_collection(COLLECTION_EMAIL)
    web_col   = client.get_collection(COLLECTION_WEB)
    print(f"Knowledge base: {email_col.count()} email pairs, {web_col.count()} web chunks")
except Exception as e:
    print(f"Error loading collections: {e}")
    print("Run app/ingest.py and app/scraper.py first")
    exit(1)

# ── Node 1: Classifier (Mistral) ──────────────────────────────────────────────
def classify_email(email: str) -> dict:
    """Mistral extracts keywords and assesses complexity from email content."""
    prompt = f"""You are an email analyst for a university admissions office.

Analyze this email and respond in exactly this format:
TOPIC: [describe the main topic in 2-4 words extracted from the email]
COMPLEXITY: [one of: simple, complex]
KEYWORDS: [3-5 specific search terms extracted directly from this email]
NEEDS_SEARCH: [yes or no]

Rules for KEYWORDS:
- Extract terms directly from the email content
- Be specific, not generic
- Focus on what the person is actually asking about
- Do not use predefined categories

Email:
{email}

Respond with ONLY the four lines above, nothing else."""

    response = critic.invoke(prompt)

    result = {
        "topic": "general inquiry",
        "complexity": "simple",
        "keywords": "",
        "needs_search": False
    }

    for line in response.strip().split("\n"):
        if line.startswith("TOPIC:"):
            result["topic"] = line.replace("TOPIC:", "").strip()
        elif line.startswith("COMPLEXITY:"):
            result["complexity"] = line.replace("COMPLEXITY:", "").strip().lower()
        elif line.startswith("KEYWORDS:"):
            result["keywords"] = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("NEEDS_SEARCH:"):
            result["needs_search"] = "yes" in line.lower()

    return result

# ── Node 2: Retriever ─────────────────────────────────────────────────────────
def retrieve_context(email: str) -> dict:
    """Search both ChromaDB collections."""
    embedding = embedder.encode(email).tolist()

    email_results = email_col.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )
    web_results = web_col.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )

    email_context = []
    for doc, meta in zip(email_results["documents"][0], email_results["metadatas"][0]):
        email_context.append({
            "question": doc,
            "answer": meta.get("answer", ""),
        })

    web_context = []
    for doc, meta in zip(web_results["documents"][0], web_results["metadatas"][0]):
        web_context.append({
            "content": doc,
            "source": meta.get("source", "website")
        })

    return {"email_context": email_context, "web_context": web_context}

# ── Node 3: Web Search ────────────────────────────────────────────────────────
def search_lafollette(keywords: str) -> list:
    """Search only within lafollette.wisc.edu."""
    query = f"{SITE_RESTRICT} {keywords}"
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append({
                    "title":   r.get("title", ""),
                    "content": r.get("body", ""),
                    "url":     r.get("href", "")
                })
        print(f"  Web search: {len(results)} results for '{keywords}'")
    except Exception as e:
        print(f"  Web search failed: {e}")
    return results

# ── Node 4: Drafter (Llama 3) ─────────────────────────────────────────────────
def draft_reply(email: str, context: dict, search_results: list,
                topic: str, critique: str = None) -> str:
    """Llama 3 drafts the reply using tone from emails, facts from web."""

    tone_parts = ["=== TONE AND FORMAT REFERENCE (style only, not facts) ==="]
    for i, item in enumerate(context["email_context"]):
        tone_parts.append(f"\nExample {i+1}:")
        tone_parts.append(f"Question: {item['question']}")
        tone_parts.append(f"Answer: {item['answer']}")

    fact_parts = ["=== FACTUAL SOURCES (use only these for any facts) ==="]
    for item in context["web_context"]:
        fact_parts.append(f"\n[{item['source']}]\n{item['content']}")
    for r in search_results:
        fact_parts.append(f"\n[{r['url']}]\n{r['content']}")
    if not context["web_context"] and not search_results:
        fact_parts.append("No factual sources available. Direct applicant to website or office.")

    revision_section = ""
    if critique:
        revision_section = f"""
=== CRITIQUE FROM REVIEWER (address all points) ===
{critique}
"""

    prompt = f"""You are an admissions assistant at the La Follette School of Public Affairs, UW-Madison.

Draft a professional, warm reply to the incoming email.

STRICT RULES:
1. Tone and format: match the past email examples
2. Facts: use ONLY the factual sources — never use facts from past email examples
3. If specific details like course names, deadlines, or requirements are NOT explicitly 
   in the factual sources, do NOT invent them — instead write exactly:
   "I will follow up with the specific details shortly" or direct them to the website
4. Never use placeholder text like [list specific courses] — either state the real fact 
   or acknowledge you will follow up
5. Never make up course names, dates, requirements, or any specific facts
6. When listing ANYTHING — courses, prerequisites, requirements, funding options, 
   electives, career options, deadlines, or any enumerable items — ALWAYS format 
   them as bullet points or numbered lists, never as inline comma-separated text.
   This includes elective courses — never list them in a single sentence separated 
   by commas, always use bullet points.
7. Write a complete email — do not leave any question unanswered if the information 
   is available in the factual sources
8. Include the relevant website link naturally at the end of the reply
9. Always sign off with exactly this format:
   "Best regards,
   [Name]
   Admissions Team
   La Follette School of Public Affairs"
   where [Name] is a placeholder the human will fill in before sending
10. Topic: {topic}
{revision_section}
{chr(10).join(tone_parts)}

{chr(10).join(fact_parts)}

=== INCOMING EMAIL ===
{email}

=== DRAFT REPLY ==="""

    return drafter.invoke(prompt)

# ── Node 5: Critic (Mistral) ──────────────────────────────────────────────────
def critique_draft(email: str, draft: str, context: dict,
                   search_results: list) -> dict:
    """Mistral critically evaluates the draft independently."""

    all_facts = []
    for item in context["web_context"]:
        all_facts.append(item["content"])
    for r in search_results:
        all_facts.append(r["content"])
    facts_str = "\n\n".join(all_facts) if all_facts else "No factual sources available."

    prompt = f"""You are a strict quality reviewer for university admissions email responses.

Critically evaluate this draft reply. Be harsh — your job is to find problems.

Check for:
1. Factual accuracy — are all facts supported by the provided sources?
2. Completeness — does it address every question asked?
3. Tone — is it professional and warm, matching a university admissions office?
4. Clarity — is it easy to understand?
5. Hallucinations — does it state anything not found in the sources?

Respond in exactly this format:
ACCURACY: [1-5]
COMPLETENESS: [1-5]
TONE: [1-5]
CLARITY: [1-5]
HALLUCINATIONS: [yes or no]
APPROVE: [yes or no — yes only if all scores are 4 or above and no hallucinations]
CRITIQUE: [specific actionable feedback, or "none" if approved]

Original email:
{email}

Draft reply:
{draft}

Verified factual sources:
{facts_str}

Respond with ONLY the seven lines above."""

    response = critic.invoke(prompt)

    result = {
        "accuracy": 3,
        "completeness": 3,
        "tone": 3,
        "clarity": 3,
        "hallucinations": False,
        "approve": False,
        "critique": "none",
        "overall": 3.0
    }

    for line in response.strip().split("\n"):
        try:
            if line.startswith("ACCURACY:"):
                result["accuracy"] = int(line.replace("ACCURACY:", "").strip()[0])
            elif line.startswith("COMPLETENESS:"):
                result["completeness"] = int(line.replace("COMPLETENESS:", "").strip()[0])
            elif line.startswith("TONE:"):
                result["tone"] = int(line.replace("TONE:", "").strip()[0])
            elif line.startswith("CLARITY:"):
                result["clarity"] = int(line.replace("CLARITY:", "").strip()[0])
            elif line.startswith("HALLUCINATIONS:"):
                result["hallucinations"] = "yes" in line.lower()
            elif line.startswith("APPROVE:"):
                result["approve"] = "yes" in line.lower()
            elif line.startswith("CRITIQUE:"):
                result["critique"] = line.replace("CRITIQUE:", "").strip()
        except:
            pass

    result["overall"] = round(
        (result["accuracy"] + result["completeness"] +
         result["tone"] + result["clarity"]) / 4, 1
    )

    return result

# ── Node 6: Judge (Mistral) ───────────────────────────────────────────────────
def final_judgement(email: str, draft: str, loop_count: int,
                    last_scores: dict) -> dict:
    """Mistral makes the final call after revision loops are exhausted."""

    prompt = f"""You are the final quality judge for university admissions email responses.

This draft has gone through {loop_count} revision(s). Make a final judgement.

Is this draft good enough to present to a human for review and sending?
Consider: would this response help the applicant and reflect well on the institution?

Respond in exactly this format:
FINAL_SCORE: [1-5 overall quality]
SEND_TO_HUMAN: [yes or no]
SUMMARY: [one sentence summary of the draft quality]

Original email:
{email}

Final draft:
{draft}

Respond with ONLY the three lines above."""

    response = critic.invoke(prompt)

    result = {
        "final_score": last_scores.get("overall", 3.0),
        "send_to_human": True,
        "summary": "Ready for human review."
    }

    for line in response.strip().split("\n"):
        try:
            if line.startswith("FINAL_SCORE:"):
                result["final_score"] = float(line.replace("FINAL_SCORE:", "").strip()[0])
            elif line.startswith("SEND_TO_HUMAN:"):
                result["send_to_human"] = "yes" in line.lower()
            elif line.startswith("SUMMARY:"):
                result["summary"] = line.replace("SUMMARY:", "").strip()
        except:
            pass

    return result

# ── Main Agent Loop ───────────────────────────────────────────────────────────
def run_agent(email: str) -> None:
    print("\n" + "="*60)
    print("MULTI-MODEL AGENTIC RAG EMAIL ASSISTANT")
    print(f"Drafter: {DRAFTER_MODEL} | Critic: {CRITIC_MODEL}")
    print("="*60)

    # Node 1: Mistral classifies
    print("\n[Node 1] Mistral classifying email...")
    classification = classify_email(email)
    print(f"  Topic:        {classification['topic']}")
    print(f"  Complexity:   {classification['complexity']}")
    print(f"  Keywords:     {classification['keywords']}")
    print(f"  Needs search: {classification['needs_search']}")

    # Node 2: Retrieve from ChromaDB
    print("\n[Node 2] Retrieving from knowledge base...")
    context = retrieve_context(email)
    print(f"  Email matches: {len(context['email_context'])}")
    print(f"  Web matches:   {len(context['web_context'])}")

    # Node 3: Web search if needed
    search_results = []
    if classification["needs_search"] and classification["keywords"]:
        print("\n[Node 3] Searching lafollette.wisc.edu...")
        search_results = search_lafollette(classification["keywords"])
    else:
        print("\n[Node 3] Web search not needed — skipping")

    # Node 4 + 5: Llama 3 drafts, Mistral critiques, loop
    critique = None
    last_scores = {}
    loop = 0

    while loop < MAX_LOOPS:
        print(f"\n[Node 4] Llama 3 drafting reply (loop {loop + 1})...")
        draft = draft_reply(
            email, context, search_results,
            classification["topic"], critique
        )

        print(f"\n[Node 5] Mistral critiquing draft (loop {loop + 1})...")
        scores = critique_draft(email, draft, context, search_results)
        last_scores = scores

        print(f"  Accuracy:       {scores['accuracy']}/5")
        print(f"  Completeness:   {scores['completeness']}/5")
        print(f"  Tone:           {scores['tone']}/5")
        print(f"  Clarity:        {scores['clarity']}/5")
        print(f"  Hallucinations: {'YES — flagged' if scores['hallucinations'] else 'None detected'}")
        print(f"  Overall:        {scores['overall']}/5")
        print(f"  Approved:       {'Yes' if scores['approve'] else 'No'}")

        if scores["approve"]:
            print("  Mistral approved the draft.")
            break

        critique = scores["critique"]
        print(f"  Critique: {critique}")
        loop += 1

    # Node 6: Mistral final judgement
    print("\n[Node 6] Mistral making final judgement...")
    judgement = final_judgement(email, draft, loop, last_scores)
    print(f"  Final score:    {judgement['final_score']}/5")
    print(f"  Send to human:  {'Yes' if judgement['send_to_human'] else 'No — needs more work'}")
    print(f"  Summary:        {judgement['summary']}")

    # Output
    print("\n" + "="*60)
    print("FINAL DRAFT")
    print("="*60)
    print(draft)
    print("\n" + "="*60)
    print(f"Overall quality:  {judgement['final_score']}/5")
    print(f"Loops completed:  {loop}")
    print(f"Judgement:        {judgement['summary']}")
    if not judgement["send_to_human"]:
        print("FLAG: Low quality — do not send without significant revision")
    elif last_scores.get("hallucinations"):
        print("FLAG: Hallucinations detected — verify all facts before sending")
    else:
        print("STATUS: Ready for human review")
    print("="*60)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Paste the incoming email below.")
    print("When done, type END on a new line and press Enter.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    email = "\n".join(lines)

    if email.strip():
        run_agent(email)
    else:
        print("No email provided.")