from __future__ import annotations
from typing import Any, Iterable, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import logging
from fastapi import HTTPException
from qdrant_client.http.exceptions import UnexpectedResponse
from utils import load_config

log = logging.getLogger(__name__)
config = load_config()

QDRANT_URL = config["qdrant_url"]
COLLECTION_NAME = config["collection"]
VECTOR_SIZE = config["vectorsize"]

# Reuse one client (fine for typical FastAPI usage)
client = QdrantClient(url=QDRANT_URL)

def ensure_collection(
    collection_name: str = COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
    distance: Distance = Distance.COSINE,
) -> None:
    
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(size = vector_size, 
                                      distance = distance)
    )


def upsert_points(
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: Optional[list[int]] = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    
    if len(vectors) != len(payloads):
        raise ValueError("vectors and payloads must have the same length")
    
    ensure_collection(collection_name = collection_name)
    
    if ids is None:
        # (Better: use UUIDs or a real ID strategy to avoid collisions.)
        ids = list(range(1, len(vectors) + 1))
    points = [
        PointStruct(id=pid, 
                    vector=vec, 
                    payload=pay)
        for pid, vec, pay in zip(ids, vectors, payloads)
    ]
    
    client.upsert(collection_name = collection_name, 
                  points = points)


def search(
    query_vector: list[float],
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
    with_payload: bool = True,
):
    try:
        ensure_collection(collection_name=collection_name)

        return client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=with_payload,
        )
    except UnexpectedResponse as e:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {str(e)}")
    

#example
# from .qdrant_store import upsert_points
# upsert_points(
#     vectors=[embedding1, embedding2],
#     payloads=[{"text": chunk1}, {"text": chunk2}],
#     # ids optional
# )