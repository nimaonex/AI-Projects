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
    
    return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

RULES:
    1. Use only the document context.
    2. Never invent information.
    3. If information is missing, say so.
    4. Quote relevant passages when useful.
    5. Cite document names.
    6. Keep answers clear and professional.

If the answer cannot be found in the provided context:
    1. Say the information is unavailable.
    2. Explain what information would be needed.
    3. Do not guess.

Context:
{context_block}

Question: 
{question}

Requirements:
- Explain your reasoning.
- Cite supporting passages.
- Highlight uncertainties.
- Distinguish between facts and assumptions.

Answer:
"""

# Chat History:
# {chat_history}