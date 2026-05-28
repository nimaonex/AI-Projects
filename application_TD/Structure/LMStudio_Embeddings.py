import requests

# Assuming LM Studio is running on the default port
def get_embedding(text):

    LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"
    
    payload = {
        "input": text,
        #"model": "text-embedding-nomic-embed-text-v1.5", # 768 dimensional embeddings 
        "model": "text-embedding-mxbai-embed-large-v1", # 1024 dimensional embeddings
    }
    response = requests.post(LM_STUDIO_URL, json=payload)
    return response.json()["data"][0]["embedding"]


# my_embedding = get_embedding("Your text to embed")
# print(len(my_embedding))