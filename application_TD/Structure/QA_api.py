from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from embedding_client import embed_texts
from qdrant_store import search, VECTOR_SIZE
from PromptBuilder import build_rag_prompt
from LLM_Client import generate
import logging
from utils import load_config
from typing import Optional

log = logging.getLogger(__name__)
config = load_config()

router = APIRouter(prefix="/qa", tags=["qa"])

class AskRequest(BaseModel):
    question: str
    top_k: int = config["top_k_documents"]
    max_tokens: int = config["max_token"]
    temperature: float = config["temperature"]
    session_id: Optional[str] = None

    
@router.post("/ask")
def ask(req: AskRequest):
       # db: Session = Depends(get_db)):  load_recent_history(db=db ...
    q = (req.question or "").strip()

    if not q:        
        log.error("question error in /qa")

        raise HTTPException(status_code=400, detail="question is required")
    
    qvec = embed_texts([q])[0]
    if len(qvec) != VECTOR_SIZE:
        log.error("vector size error in /qa")

        raise HTTPException(
            status_code=500,
            detail=f"Embedding size {len(qvec)} != expected {VECTOR_SIZE}",
        )
    hits = search(query_vector=qvec, 
                  top_k=req.top_k)

    contexts = []
    for h in hits.points:
        payload = h.payload or {}
        contexts.append({
            "score": float(h.score),
            "text": payload.get("text", ""),
            "filename": payload.get("filename", ""),
            "chunk_index": payload.get("chunk_index", None),
        })

    prompt = build_rag_prompt(question=q, 
                              contexts=contexts)
    answer = generate(prompt, 
                      max_tokens=req.max_tokens, 
                      temperature=req.temperature)

    return {
        "question": q,
        "answer": answer,
        #"contexts": contexts,  # return for debugging; you can remove in prod
    }