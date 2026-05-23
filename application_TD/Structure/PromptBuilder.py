from __future__ import annotations
import logging

log = logging.getLogger(__name__)

def build_rag_prompt(question: str, contexts: list[dict]) -> str:

    """
    contexts: list of dicts like:
      {"text": "...", "filename": "...", "chunk_index": 0, "score": 0.78}
    """
    ctx_lines = []

    for i, c in enumerate(contexts, start=1):
        filename = c.get("filename", "unknown")
        chunk_index = c.get("chunk_index", "?")
        text = c.get("text", "")
        ctx_lines.append(f"[{i}] ({filename} chunk {chunk_index})\n{text}")
    context_block = "\n\n".join(ctx_lines)
    
    return f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the context is insufficient, say: "I don't know based on the provided documents."
Context:
{context_block}
Question: {question}
Answer:
"""