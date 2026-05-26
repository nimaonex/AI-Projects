from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from utils import load_config

log = logging.getLogger(__name__)
config = load_config()

load_dotenv(override=True)

def generate(prompt: str, max_tokens: int = config["max_token"], temperature: float = config["temperature"]):

    client = OpenAI(
                    base_url = config["LLM_URL"],
                    api_key = os.getenv("API_KEY")
                    ) 
    
    try:
        response = client.chat.completions.create(
                    model=config["ai_model"],
                    messages=[
                        #{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    max_tokens= max_tokens,
                    temperature= temperature
                )

        print("Model response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Error communicating with model server: {e}")