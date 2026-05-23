from langchain_text_splitters.character import CharacterTextSplitter
import logging
from utils import load_config

log = logging.getLogger(__name__)
config = load_config()

def chunk_text(text: str, chunk_size: int = config["chunk_size"], chunk_overlap: int = config["chunk_overlap"]):
    document_splitter = CharacterTextSplitter(separator = ".",
                                              chunk_size = chunk_size,
                                              chunk_overlap = chunk_overlap)

    chunked_text = document_splitter.split_text(text)

    return chunked_text