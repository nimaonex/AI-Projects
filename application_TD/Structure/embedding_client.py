from __future__ import annotations
import requests
import logging
from utils import load_config


log = logging.getLogger(__name__)
config = load_config()

EMBED_URL = config["embedding_URL"]
TIMEOUT_SEC = 60

def embed_texts(texts: list[str]) -> list[list[float]]:

    resp = requests.post(EMBED_URL, 
                         json={"texts": texts}, 
                         timeout=TIMEOUT_SEC)

    resp.raise_for_status()
    data = resp.json()
    # expects: {"embeddings": [[...], [...], ...]}
    return data["embeddings"]