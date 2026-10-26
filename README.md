# RAG Pipeline: Chat With Your Documents

A small Retrieval-Augmented Generation system that answers questions **only from your own documents**, cites sources (file + page), and **refuses to answer when nothing relevant is found**.

## Architecture
```
docs/ (PDF, TXT, MD)
   -> chunking (400 chars, 80 overlap, sentence-aware, page metadata kept)
   -> embeddings (all-MiniLM-L6-v2, local via sentence-transformers)
   -> ChromaDB (cosine similarity)
question -> embed -> top-6 chunks -> relevance gate (distance <= MAX_DISTANCE)
   -> prompt (context only) -> Groq LLM (openai/gpt-oss-120b) -> answer + sources
```

## Avoiding hallucination (two layers)
1. **Relevance gate:** chunks with cosine distance above `MAX_DISTANCE` are dropped. If none remain, the LLM is **never called** and a fixed "couldn't find anything relevant" message is returned.
2. **Strict prompt:** the model may only use the numbered context, must cite `[n]`, and must say it lacks information if the context doesn't cover the question.

## Setup
```bash
git clone <your-repo-url> && cd rag-pipeline
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add your GROQ_API_KEY (free at console.groq.com)
```

## Run
```bash
python rag.py ingest                                  # index ./docs
python rag.py ask "How many days of annual leave do employees get?"
python rag.py chat
python rag.py ask "Who won the World Cup in 2022?" --debug   # refuses; shows distances
```
Add your own PDFs / .txt / .md files to `docs/` and re-run `ingest`.

## Google Colab
Upload `rag.py`, install requirements, store `GROQ_API_KEY` in the Colab Secrets panel (or paste it with `getpass`), upload docs into `docs/`, then:
```python
import rag
rag.ingest()
rag.print_result("your question")
```

## Tuning
All settings are constants at the top of `rag.py`.
- `MAX_DISTANCE`: lower = stricter. Run with `--debug` to see distances for relevant vs. unrelated questions, then set the threshold between them.
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: smaller chunks give sharper retrieval; larger chunks give more context.
- `LLM_MODEL`: Groq retires models over time (llama-3.3-70b-versatile was shut down on 16 Aug 2026). Check https://console.groq.com/docs/models and override via the `LLM_MODEL` env var.

## Limitations
- Scanned PDFs (images only) have no extractable text; use OCR first.
- The distance threshold depends on the embedding model and your documents, so re-tune it for new data.
