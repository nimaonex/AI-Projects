from fastapi import (
     APIRouter, 
     UploadFile, 
     File, 
     HTTPException
)
from loader_class import get_loader
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
    file: UploadFile = File(...)
):

    file_bytes: bytes = await file.read()
    file_name: str = file.filename

    try:

        loader = get_loader(
             filename = file_name,
             file_content = file_bytes
            )
        
        paragraphs = loader.load()
        list_completions = [a.page_content for a in paragraphs]
        text = "\n".join(list_completions)

        chunks = chunk_text(
            text, 
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )

        if not chunks:
            log.error("chunk error in /upload_api")
            raise HTTPException(status_code=400, detail="No text/chunks extracted")
        
        # Call embedding server (batch)
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

        upsert_points(
            vectors=vectors, 
            payloads=payloads
        )

        return {"filename": file.filename, "chunks": len(chunks)}
    
    except OSError:
        log.exception("unhandled error in /upload_api")
        pass
