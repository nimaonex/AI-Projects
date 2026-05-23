from utils import load_config
from fastapi import FastAPI, Body, Request, Depends
import string
from tiktoken import encoding_for_model
from file_loader import UserQuestion
from typing import Annotated
import uuid
import time
from starlette.responses import StreamingResponse
from vector_database import load_vector_db, initialize_similarity_search
from prompt_engineering import user_message_setter
from openai import AsyncOpenAI
import json

#########################################################################################################################
config = load_config()
model_app = FastAPI()
ask_time = time.time()
#########################################################################################################################
#this function check if the question is empty or not
def check_user_question(user_question: str) -> bool:
    # Remove punctuation from the question
    translator = str.maketrans('', '', string.punctuation)
    cleaned_question = user_question.translate(translator)

    # Split the cleaned question into words
    words_len = len(cleaned_question.split())
    letters_len = len(cleaned_question)

    # Check if the number of words is 2 or more
    if words_len < config.get("min_word_to_answer", 1) or letters_len < config.get("min_char_to_answer", 0):
        return False

    return True
#########################################################################################################################
def count_tokens(messages, model):
    """
    Count the number of tokens in the given messages for a specific model.
    If counting tokens using `encoding_for_model` fails, it falls back to counting words by splitting the content.
    If both methods fail, it returns 0.

    Args:
        messages (list): List of message dictionaries.
        model (str): The model name to use for encoding.

    Returns:
        int: The number of tokens in the messages or 0 if counting fails.
    """
    try:
        # Try to load the model encoding method
        enc = encoding_for_model("gpt-4o")
    except Exception as e:
        print(f"Token encoding for model '{model}' failed: {e}. Falling back to word count.")
        enc = None  # Fallback to None to signify using word count method

    total_tokens = 0

    for message in messages:
        content = message.get("content", "")

        if enc:
            try:
                # Try counting tokens using the encoding method
                total_tokens += len(enc.encode(content))
            except Exception as e:
                print(f"Encoding failed for message '{content}': {e}. Using word count instead.")
                total_tokens += len(content.split())  # Fallback to word count
        else:
            # If encoding is not available, use word count directly
            total_tokens += len(content.split())

    return total_tokens or 0  # Return 0 if the total_tokens is still 0
#########################################################################################################################
if config["chat_memory"] == "basic":
    chat_memory = {}
# it means: the program uses a simple memory system represented by an empty dictionary. This dictionary will probably store messages, history, or conversation state.

process_query_responses = {
    200: {"description": "Success"},
    400: {"description": "Invalid request"},
}


#########################################################################################################################
#FastAPI route decorator
@model_app.post('/process_question_api', responses = process_query_responses)
async def process_user_question_api(question: UserQuestion,
                                    #company_id: Annotated[int, Body(description="Enter the company name here")],
                                    request: Request,
                                    #is_private: Annotated[int, Body(description="1 If The User Want To See Their Companies Document")] = 1,
                                    #is_test: Annotated[bool, Body(description="Is the request for dev purposes?")] = False,
                                    user_id: Annotated[str, Body(description="User's Session ID To Recieve User's Chat History")] = str(uuid.uuid4()),
                                    #username: str = Depends(basic_auth),
                                    ):
    user_text_question = question.user_text_question
    user_question = user_text_question
    stream = question.stream

    user_token = user_id
    if config["chat_memory"] == "basic":
        if user_token not in chat_memory:
            chat_memory[user_token] = []
        history_len = len(chat_memory[user_token])
        chat_history_lst = chat_memory[user_token]
        chat_history = str(chat_history_lst[:5])


    # Get the raw JSON payload
    payload = await request.json()


    # Document retrieval
    async def get_document_data():
        if config["db_type"] == "chroma":
            db = load_vector_db(
                #company_id=company_id,
                db_type=config["db_type"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
            ) 
            print('db load time', time.time() - ask_time)
            return initialize_similarity_search(
                #company_id=company_id,
                db=db,
                query=user_question,
                #is_private=is_private
            )
    
    doc_result = await get_document_data()
    prob_docs, sources, scores = doc_result
    sources = list(set(sources))

    ai_model = "qwen2.5-3b-instruct"
    async def generate_responses():
        temperature = config["temperature"]
        user_message = user_message_setter(prob_docs, chat_history_lst, user_question)

        client = AsyncOpenAI(api_key="not-needed",
                             base_url="http://172.0.0.1:1234/v1")


        messages = [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content":user_message},
        ]

        completion = await client.chat.completions.create(
                model=ai_model,
                messages=messages,
                temperature=temperature,
                stream=True)
        
        output_tokens = []
        answer = ""

        async for token in completion:
                    event_text = token.choices[0].delta  # Access the first choice's delta
                    if event_text and event_text.content:
                        output_tokens.append(event_text.content)
                        answer += event_text.content
                        end_time = time.time()
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": ai_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event_text.content},
                                    "finish_reason": None,
                                    "content_type": "answer"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
        
        chat_history_dict = {"user": user_question, "assistant": answer}

        # if config["chat_memory"] == "redis":
        #     redis_client.rpush(history_key, json.dumps({"user": user_question, "assistant": answer}))
        if config["chat_memory"] == "basic":
            chat_memory[user_token].append(chat_history_dict)
                    
        ai_response = {
            "ai_answer": answer,
            "input_tokens": count_tokens(messages, ai_model),
            "output_tokens": len(output_tokens)
            }
        
        # Calculate processing time and print it
        process_question_time = time.time()
        print('total process question time', process_question_time - ask_time)
        # Send the sources one by one as separate chunks (if any)
    if stream:
        return StreamingResponse(generate_responses(), media_type="text/event-stream")
    else:
        return StreamingResponse(generate_responses(), media_type="application/json")    
#########################################################################################################################                
if __name__ == "__main__":
    import uvicorn

    worker_count = config.get("worker_count", 1)
    if worker_count == 1:
        uvicorn.run("model:model_app", host="0.0.0.0", port=config.get("server_port", 8765))
    else:
        uvicorn.run("model:model_app", host="0.0.0.0", port=config.get("server_port", 8765), workers=worker_count,
                    reload=False)