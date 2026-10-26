"""
RAG pipeline: chunk -> embed -> store (ChromaDB) -> retrieve -> grounded answer with sources.

    python rag.py ingest             # index everything inside ./docs
    python rag.py ask "question"     # one question
    python rag.py chat               # interactive
    add --debug to see retrieval distances (useful for tuning MAX_DISTANCE)
"""
import os
import re
import sys
from pathlib import Path

import chromadb
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ---------------- Config ----------------
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "my_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"      # local + free
CHUNK_SIZE = 400                       # characters
CHUNK_OVERLAP = 80
TOP_K = 6
# Cosine distance: 0 = identical, ~1+ = unrelated. Chunks further than this are discarded.
# Tuned on sample docs: relevant ~0.3-0.75, unrelated ~1.0. Re-tune on your own docs with --debug.
MAX_DISTANCE = 0.8
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

NO_ANSWER = ("I couldn't find anything relevant to that in your documents, "
             "so I can't answer it reliably.")

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Use only facts found in the context. Do not use outside knowledge.
- Cite passages ONLY in the plain format [1], [2] (no other citation style).
- If the context does not contain the answer, reply exactly: "I don't have enough information in the provided documents to answer that."
- If the context only partly answers the question, answer the part you can and say what is missing."""


# ---------------- Loading ----------------
def load_documents(folder: Path):
    """Yield (text, source, page) for each PDF page / text file."""
    for path in sorted(folder.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield text, path.name, i
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield text, path.name, 1


# ---------------- Chunking ----------------
def chunk_text(text: str, size=None, overlap=None):
    """Overlapping chunks, preferring to cut at sentence boundaries."""
    size = size or CHUNK_SIZE
    overlap = overlap if overlap is not None else CHUNK_OVERLAP
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end),
                      text.rfind("! ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 30]


# ---------------- Vector store ----------------
def get_client():
    return chromadb.PersistentClient(path=DB_DIR)

def embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

def ingest():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        raise SystemExit(f"Put PDFs / .txt / .md files in ./{DOCS_DIR} first.")
    client = get_client()
    try:
        client.delete_collection(COLLECTION)      # rebuild from scratch each time
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=embedder(),
                                   metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for text, source, page in load_documents(DOCS_DIR):
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}-p{page}-c{j}")
            docs.append(chunk)
            metas.append({"source": source, "page": page})
    if not ids:
        raise SystemExit("No text found in docs/ (scanned PDFs have no extractable text).")
    for i in range(0, len(ids), 100):
        col.add(ids=ids[i:i+100], documents=docs[i:i+100], metadatas=metas[i:i+100])
    print(f"Indexed {len(ids)} chunks from {DOCS_DIR}/")


# ---------------- Retrieval ----------------
def retrieve(question: str, k=TOP_K, debug=False):
    col = get_client().get_collection(COLLECTION, embedding_function=embedder())
    res = col.query(query_texts=[question], n_results=k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if debug:
            print(f"  [debug] dist={dist:.3f} {meta['source']} p{meta['page']}: {doc[:70]}...")
        if dist <= MAX_DISTANCE:                  # relevance gate
            hits.append({"text": doc, "source": meta["source"],
                         "page": meta["page"], "distance": dist})
    return hits


# ---------------- Generation ----------------
def answer(question: str, debug=False):
    hits = retrieve(question, debug=debug)
    if not hits:                                  # nothing relevant -> never call the LLM
        return NO_ANSWER, []

    key = "".join(ch for ch in os.environ.get("GROQ_API_KEY", "").strip() if ord(ch) < 128)
    if not key:
        raise SystemExit("GROQ_API_KEY is not set.")

    context = "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1))
    r = requests.post(
        GROQ_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60,
        json={"model": LLM_MODEL, "temperature": 0, "max_tokens": 800,
              "messages": [
                  {"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]})
    if not r.ok:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"]
    # some models emit citations like 【1】 or 【1†L1-L2】; normalise them to [1]
    text = re.sub(r"【(\d+)[^】]*】", r"[\1]", text)
    return text, hits


def print_result(question: str, debug=False):
    text, hits = answer(question, debug)
    print(f"\nQ: {question}\nAnswer:\n{text}\n")
    if hits:
        print("Sources:")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] {h['source']} (page {h['page']}) - similarity {1 - h['distance']:.2f}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    if not args:
        sys.exit(__doc__)
    if args[0] == "ingest":
        ingest()
    elif args[0] == "ask" and len(args) > 1:
        print_result(" ".join(args[1:]), debug)
    elif args[0] == "chat":
        print("Ask about your documents (type 'exit' to quit)")
        while True:
            q = input("You: ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            print_result(q, debug)
    else:
        sys.exit(__doc__)
