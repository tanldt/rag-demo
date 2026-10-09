#!/usr/bin/env python3
"""
Minimal RAG demo: chunk -> embed -> retrieve -> generate
=============================================================
Run:  .venv/bin/python rag_demo.py "What is chunking?"
Options: --top-k 3   (number of chunks to retrieve)
         --rebuild   (recompute embeddings and ignore cache)

Pipeline:
  1. Read .txt documents from the docs/ directory
  2. Split each document into small overlapping chunks
  3. Convert each chunk into an embedding vector with sentence-transformers
  4. Embed the question and find the top-k closest chunks with cosine similarity
  5. Combine the chunks and question into a prompt
  6. Call an LLM for the answer (local Ollama -> OpenAI -> dry-run prompt)
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CACHE_DIR = BASE_DIR / ".cache"
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, lightweight, free, local


# ---------------------------------------------------------------- 1. Read documents
def load_documents():
    """Read all .txt files in docs/ -> list of (file_name, content)."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


# ---------------------------------------------------------------- 2. Chunking
def chunk_text(text, chunk_size=60, overlap=15):
    """Split text into word-based chunks with `overlap` shared words.

    Example: chunk_size=60, overlap=15 -> chunk 2 starts at word 46 of chunk 1,
    so ideas split across boundaries are less likely to be lost.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap  # step back to create overlap
    return chunks


# ---------------------------------------------------------------- 3. Embedding
def get_model():
    """Load the embedding model (first run downloads ~90MB, later runs use cache)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_chunks(all_chunks, rebuild=False):
    """Convert chunks into vectors and cache them for future runs."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "embeddings.npz"

    if cache_file.exists() and not rebuild:
        data = np.load(cache_file, allow_pickle=True)
        embeddings = data["embeddings"]
        if len(embeddings) == len(all_chunks):
            print(f"  (using embedding cache: {len(embeddings)} chunks)")
            return embeddings
        print(
            f"  (ignoring cache: found {len(embeddings)} embeddings "
            f"but currently have {len(all_chunks)} chunks)"
        )

    model = get_model()
    print(f"  embedding {len(all_chunks)} chunks with {EMBED_MODEL} ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    np.savez(cache_file, embeddings=embeddings,
             chunks=np.array(all_chunks, dtype=object))
    return embeddings


# ---------------------------------------------------------------- 4. Retrieval
def cosine_similarity(a, b):
    """Compute cosine similarity between vector a and matrix b (one vector per row)."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm  # dot product equals cosine after normalization


def retrieve(question, chunks, embeddings, model, top_k=3):
    """Embed the question and return the top-k closest chunks with similarity scores."""
    q_emb = model.encode([question])[0]
    scores = cosine_similarity(q_emb, embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------- 5. Build prompt
def build_prompt(question, results):
    """Combine numbered chunks and the question into a prompt."""
    context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, (chunk, _) in enumerate(results)
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the
context below. If the answer is not in the context, say you don't know.
Cite the chunk numbers you used, like [1], [2].

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------- 6. Generate
def generate_with_ollama(prompt, model_name="llama3"):
    """Call local Ollama (http://localhost:11434). Install from https://ollama.com."""
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(
            {"model": model_name, "prompt": prompt, "stream": False}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def generate_with_openai(prompt):
    """Call the OpenAI API. Requires the OPENAI_API_KEY environment variable."""
    from urllib.request import Request, urlopen

    key = os.environ["OPENAI_API_KEY"]
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def generate(prompt):
    """Try Ollama -> OpenAI -> dry-run (print a prompt to paste into any LLM)."""
    try:
        print("  (calling local Ollama ...)")
        return generate_with_ollama(prompt)
    except Exception as e:
        print(f"  (Ollama unavailable: {type(e).__name__})")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            print("  (calling OpenAI ...)")
            return generate_with_openai(prompt)
        except Exception as e:
            print(f"  (OpenAI error: {e})")
    print("\n  [!] No LLM found. Dry-run mode: use the prompt below,")
    print("      and paste it into ChatGPT, Claude, or another LLM.\n")
    return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Minimal RAG demo")
    ap.add_argument("question", help="question in English, wrapped in quotes")
    ap.add_argument("--top-k", type=int, default=3, help="number of chunks to retrieve")
    ap.add_argument("--rebuild", action="store_true", help="recompute embeddings")
    args = ap.parse_args()

    print("\n[1/6] Reading documents ...")
    docs = load_documents()
    print(f"  read {len(docs)} files")

    print("[2/6] Chunking ...")
    chunks, sources = [], []
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    print(f"  created {len(chunks)} chunks")

    print("[3/6] Embedding ...")
    embeddings = embed_chunks(chunks, rebuild=args.rebuild)

    print("[4/6] Retrieval ...")
    model = get_model()
    results = retrieve(args.question, chunks, embeddings, model, args.top_k)
    for i, (chunk, score) in enumerate(results):
        src = sources[chunks.index(chunk)]
        print(f"  #{i + 1} score={score:.3f} ({src}): {chunk[:80]}...")

    print("[5/6] Building prompt ...")
    prompt = build_prompt(args.question, results)

    print("[6/6] Generate ...")
    answer = generate(prompt)
    if answer:
        print("\n===== ANSWER =====\n")
        print(answer)
    else:
        print("===== PROMPT (copy into an LLM) =====\n")
        print(prompt)


if __name__ == "__main__":
    main()
