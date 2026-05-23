import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from DocumentLoader import DOCXLoader
from chunker import chunk_text
from embedding_client import embed_texts
from qdrant_store import upsert_points, VECTOR_SIZE
import logging
from utils import load_config

log = logging.getLogger(__name__)
config = load_config()
 
router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("")
async def ingest_docx(
    file: UploadFile = File(...),
    #chunk_size: int = Form(config["chunk_size"]),
    #overlap: int = Form(config["chunk_overlap"]),
):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx is supported")
    
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        #tmp.write(await file.read())

    file_bytes: bytes = await file.read()

    try:
        #text = DOCXLoader().load_text(tmp_path)
        loader = DOCXLoader(file_content= file_bytes)
        paragraphs = loader.load()
        list_completions = [a.page_content for a in paragraphs]
        text = "\n".join(list_completions)
        chunks = chunk_text(text, chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])

        if not chunks:
            log.error("chunk error in /upload_api")
            raise HTTPException(status_code=400, detail="No text/chunks extracted")
        
        # Call your embedding server (batch)
        vectors = embed_texts(chunks)
        # Quick sanity check: embedding dim should be 1024 (per your config)
        if len(vectors[0]) != VECTOR_SIZE:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding size {len(vectors[0])} != expected {VECTOR_SIZE}"
            )
        payloads = [
            {"filename": file.filename, 
             "chunk_index": i, 
             "text": ch}

            for i, ch in enumerate(chunks)
        ]

        upsert_points(vectors=vectors, payloads=payloads)

        return {"filename": file.filename, "chunks": len(chunks)}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            log.exception("unhandled error in /upload_api")
            pass