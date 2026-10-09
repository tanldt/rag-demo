# Minimal RAG Demo

A small end-to-end RAG learning project: **chunk -> embed -> retrieve -> generate**. It runs locally on your machine, except for the optional LLM call. If no LLM is configured, it falls back to dry-run mode and prints the prompt.

## Setup

```bash
cd ~/rag-demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> No GPU: install the lighter CPU build of torch first:
> `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`
> then install the rest of `requirements.txt`.

The first run downloads the embedding model `all-MiniLM-L6-v2` (~90MB). Later runs use the local cache.

## Try It

```bash
.venv/bin/python rag_demo.py "What is chunking?"
.venv/bin/python rag_demo.py "What is the difference between IVF and HNSW?" --top-k 3
.venv/bin/python rag_demo.py "How do I evaluate a RAG system?" --rebuild
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe rag_demo.py "What is chunking?"
.\.venv\Scripts\python.exe rag_demo.py "What is the difference between IVF and HNSW?" --top-k 3
.\.venv\Scripts\python.exe rag_demo.py "How do I evaluate a RAG system?" --rebuild
```

You will see six steps printed: reading documents -> chunking -> embedding -> retrieval with similarity scores -> prompt building -> generation.

## 3 Generation Modes

1. **Ollama (recommended, free):** install it from https://ollama.com, then run `ollama pull llama3`. The script detects it and calls it locally.
2. **OpenAI:** set `OPENAI_API_KEY`, then run the script. It uses `gpt-4o-mini`.
3. **Dry-run:** if neither Ollama nor OpenAI is available, the script prints the full prompt so you can paste it into ChatGPT, Claude, or another LLM.

## Structure

```
rag-demo/
├── docs/               # sample RAG documents; add your own .txt files here
├── rag_demo.py         # the full pipeline with step-by-step comments
├── requirements.txt
└── .cache/             # computed embeddings; delete or use --rebuild to recompute
```

## Experiments To Try

1. Change `chunk_size` or `overlap` in `chunk_text()` and compare retrieval scores.
2. Compare `--top-k 1` with `--top-k 5` and see how the answer changes.
3. Ask a question that is not covered by the documents, such as "What is the capital of France?", and check whether the model says it does not know.
4. Add your own `.txt` file to `docs/`, run with `--rebuild`, and ask about that content.
5. Try another embedding model, such as `all-mpnet-base-v2`, which is more accurate but slower.

## Next Steps

- Replace numpy brute-force search with **ChromaDB**, a real local vector database.
- Try **hybrid search** by combining vector similarity with keyword search (BM25).
- Add a **reranker** after retrieval.
- Package the project as a FastAPI service with a simple chat UI.
