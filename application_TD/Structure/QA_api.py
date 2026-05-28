from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from embedding_client import embed_texts
from qdrant_store import search, VECTOR_SIZE
from PromptBuilder import build_rag_prompt
from LLM_Client import generate
import logging
from utils import load_config
from typing import Optional
from chat_history import (
    save_message,
    load_summary,
    load_recent_messages,
    rewrite_question,
    summarize_conversation,
    save_summary,
    get_db
)
from sqlalchemy.orm import Session



log = logging.getLogger(__name__)
config = load_config()

router = APIRouter(prefix="/qa", tags=["qa"])

class AskRequest(BaseModel):
    question: str
    top_k: int = config["top_k_documents"]
    max_tokens: int = config["max_token"]
    temperature: float = config["temperature"]
    sessionid: str = None

    
@router.post("/ask")
def ask(req: AskRequest,
        sessionid: AskRequest,
        db: Session = Depends(get_db)):

    try:

        q = (req.question or "").strip()

        if not q:        
            log.error("question error in /qa")

            raise HTTPException(status_code=400, detail="question is required")
        
        save_message(
                db= db, 
                session_id=sessionid, 
                role="user", 
                content=q
                )
        
        summary = load_summary(
                db= db, 
                session_id=sessionid
                )
        
        history = load_recent_messages(
                db= db, 
                session_id=sessionid
                )
        
        Modified_question = rewrite_question(question=q, 
                                            history=history, 
                                            summary=summary
                                            )

        qvec = embed_texts([Modified_question])[0]

        if len(qvec) != VECTOR_SIZE:
            log.error("vector size error in /qa")

            raise HTTPException(
                status_code=500,
                detail=f"Embedding size {len(qvec)} != expected {VECTOR_SIZE}",
            )
        
        hits = search(
                    query_vector=qvec, 
                    top_k=req.top_k
                    )

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
                                contexts=contexts
                                )
        
        answer = generate(prompt, 
                        max_tokens=req.max_tokens, 
                        temperature=req.temperature
                        )
        
        save_message(
                db= db, 
                session_id=sessionid, 
                role="system", 
                content=answer
                )
        
        new_summary = summarize_conversation(
                db= db, 
                session_id=sessionid
                )
        
        save_summary(
                db= db, 
                session_id=sessionid,
                summary_text=new_summary
                )
        

        db.commit()
        db.close()


        return {
            "question": q,
            "answer": answer,
        }
    except Exception:

        db.rollback()
        
        raise