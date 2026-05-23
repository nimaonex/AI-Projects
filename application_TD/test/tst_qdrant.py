import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance, PointStruct


#get embedding
# embedding url: "Snowflake/snowflake-arctic-embed-l-v2.0"
resp = requests.post(
    "http://172.17.11.82:8000/embed/dense",
    json={"texts": ["This is a test sentence"]}
)
embedding = resp.json()["embeddings"][0]

# save in qdrant vector database
client = QdrantClient(url="http://localhost:6333")
if client.collection_exists("nima-test"):
    pass
else:
    client.create_collection(collection_name = "nima-test",
                             vectors_config = VectorParams(size = 1024,
                                                           distance = Distance.COSINE))
client.upsert(
    collection_name = "nima-test" ,
    points=[
        PointStruct(
            id=1,
            vector=embedding,
            payload={"text": "This is a test sentence"}
        )
    ]
) 
#######################################################################################################
# If this does not raise an exception, Qdrant is running
collections = client.get_collections()
print("Qdrant is running ✅") 

#list of all collections
for i in client.get_collections().collections:
    print(i.name)

#collection info
c_name = "nima-test2"
info = client.get_collection(c_name)

print(f"Status: {info.status}")               # green / yellow
print(f"Points count: {info.points_count}")
print(f"Vectors config: {info.config.params.vectors}")

#see what is inside
points, _ = client.scroll(
    "nima-test2",
    limit=10,
    with_payload=True,
    with_vectors=True
)
for p in points:
    print(p) 