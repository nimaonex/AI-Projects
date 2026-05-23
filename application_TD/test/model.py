import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_text_splitters.character import CharacterTextSplitter


document_splitter = CharacterTextSplitter(separator = ".",
                                          chunk_size = 500,
                                          chunk_overlap = 50)

url = "http://127.0.0.1:8000/upload-docx"

files = {"file": open(r"E:\AI Projects\doc_files\CE1_EN.docx", "rb")}
response = requests.post(url, files=files)
paragraph = response.json()["content"]

text = "\n".join(paragraph)
chunked_text = document_splitter.split_text(text)

client = QdrantClient(url="http://localhost:6333")

client.delete_collection("assessment")

if client.collection_exists("assessment"):
    pass
else:
    client.create_collection(collection_name = "assessment",
                             vectors_config = VectorParams(size = 1024,
                                                           distance = Distance.COSINE))

for i, chunk in enumerate(chunked_text):
    resp = requests.post(
    "http://172.17.11.82:8000/embed/dense",
    json={"texts": [chunk]})

    embedding = resp.json()["embeddings"][0]

    client.upsert(
        collection_name = "assessment" ,
        points=[
            PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk}
            )
        ]
    ) 