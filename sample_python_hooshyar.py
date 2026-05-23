from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv(override=True)

client = OpenAI(
    base_url = "http://172.30.14.32:5000/v1",
    api_key = os.getenv("API_KEY")
    ) 

try:
    response = client.chat.completions.create(
                model="gemma-3-pass",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "give me just a simple list of middle east countries"}
                ],
                stream=False
            )

    print("Model response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Error communicating with model server: {e}")
