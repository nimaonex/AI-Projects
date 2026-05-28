from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

def generate(prompt: str, max_tokens: int = 400, temperature: float = 0.2):

    client = OpenAI(
                    base_url = "http://localhost:1234/v1",
                    api_key = "not-needed"
                    ) 
    
    try:
        response = client.chat.completions.create(
                    model="google/gemma-3-4b",
                    messages=[
                        #{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    max_tokens= max_tokens,
                    temperature= temperature
                )

        #print("Model response:")
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error communicating with model server: {e}")