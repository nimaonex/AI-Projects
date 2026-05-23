import requests
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = QdrantClient(url="http://localhost:6333")
client_opeai = OpenAI(
        base_url = "http://172.30.14.32:5000/v1",
        api_key = os.getenv("API_KEY")
        )

def embed(question: str) -> list[float]:
    resp = requests.post(
    "http://172.17.11.82:8000/embed/dense",
    json={"texts": [question]})
    
    return resp.json()["embeddings"][0]

def retrieve_question(question: str, collection: str,  k: int = 3) -> list[str]:   

    embedding = embed(question)

    results = client.query_points(
        collection_name=collection,
        query=embedding,
        limit=k
    )

    return [i.payload["text"] for i in results.points]

def RAG_LLM(question: str, collection: str,  k: int = 3) -> str:
    
    context = "\n".join(retrieve_question(question, collection, k))
    prompt = f"""
                    You are a document QA assistant.

                    Carefully analyze the retrieved document sections before answering.

                    Rules:
                    1. Answer strictly from the context.
                    2. If information is incomplete, explicitly mention it.
                    3. Combine information across sections when necessary.
                    4. Cite section numbers or filenames if available.
                    5. Avoid generic explanations unless supported by context.

                    Context:
                    {context}

                    Question:
                    {question}
                    """
    try:
        response = client_opeai.chat.completions.create(
                    model="gemma-3-pass",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )

        return print(response.choices[0].message.content)

    except Exception as e:
        print(f"Error communicating with model server: {e}")


question = "tell me about environment"
collection = "assessment"

RAG_LLM(question, collection) 