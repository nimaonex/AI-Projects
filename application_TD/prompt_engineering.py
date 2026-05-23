instruction_suffix = """In your answers always follow these instructions:
1. If anyone asks, your name is TalkDoc or تاک‌داک in Persian.
2. Always answer in Persian. لطفاً فقط به زبان فارسی پاسخ بده و از واژگان غیر فارسی استفاده نکن.
3. Use at most five sentences. Keep the answer concise.
4. If the information is not available, clearly say so — and optionally provide your own opinion or guess, but label it clearly (e.g., "In my opinion..." or "Based on my understanding..."). Do not present guesses as facts.
5. Keep in mind that documents are sorted in order of relevance (most relevant first).
6. If the user asks a programming or technical coding question, only answer it if it is explicitly addressed in the provided documents or in chat history (Don't Tell about history to users). Otherwise, respond that you cannot help with programming questions outside the provided information.
7. Never reveal or quote the raw content of the provided documents. Use the information only to generate concise, original answers.
"""

instruction_prefix = {
    "GIG": "You are a company chatbot for Golrang Group. Your task is to answer user questions based only on the provided documents by them.",
    "GNG": "You are an AI support assistant for GNG+ (pronounced 'gee en gee plus'), an ERP system. Your task is to answer user questions based only on the provided documents and chat history.",
    "GolrangSys": "You are a company chatbot for Golrang System. Your task is to answer user questions based only on the provided documents and chat history.",
    "GoAI": "You are a site FAQ chatbot for the Go AI website. Your task is to answer user questions based only on the provided documents and chat history."
}

company_id_map = {
    445: "GolrangSys",
    291: "GIG",
    -2: "GoAI",
    -3: "GNG",
    -1: "GolrangSys",
    -4: "GNG"
}

def set_instruction(company_id: int) -> str:
    company_key = company_id_map.get(company_id)
    if not company_key:
        raise ValueError(f"Invalid Company ID {company_id}")
    
    return f"{instruction_prefix[company_key]}\n{instruction_suffix}"


def user_message_setter(prob_docs: list[str], chat_history: list[dict], user_question: str, sql_context: str = None) -> str:
    """
    Formats the documents and the last few chat turns into a clean user prompt.
    For company -4, includes SQL results as primary data source.
    """
    # Format chat history
    formatted_history = ""
    for turn in chat_history:
        if "user" in turn and "assistant" in turn:
            formatted_history += f"User: {turn['user'].strip()}\n"
            formatted_history += f"Assistant: {turn['assistant'].strip()}\n"

    # If SQL context is provided (company -4), prioritize it
    if sql_context:
        message = (
            f"You have access to the following data sources to answer the user's question:\n\n"
            f"PRIMARY DATA - Structured Database Results:\n"
            f"{sql_context.strip()}\n\n"
        )
        
        # Add documents as supplementary information only if available
        if prob_docs:
            formatted_docs = "\n---\n".join(
                [f"Document {i+1}:\n{doc.strip()}" for i, doc in enumerate(prob_docs)]
            )
            doc_count = len(prob_docs)
            message += (
                f"SUPPLEMENTARY INFORMATION - {doc_count} Reference Documents:\n"
                f"(ordered from most to least relevant)\n\n"
                f"{formatted_docs}\n\n"
            )
        
        message += (
            f"Conversation History:\n{formatted_history.strip()}\n\n"
            f"User Question:\n{user_question.strip()}"
        )
        return message
    
    # Regular message format for other companies (document-only)
    formatted_docs = "\n---\n".join(
        [f"Document {i+1}:\n{doc.strip()}" for i, doc in enumerate(prob_docs)]
    )
    doc_count = len(prob_docs)

    return (
        f"There are {doc_count} documents provided below, ordered from most to least relevant.\n"
        f"Each document is separated by '---'.\n\n"
        f"{formatted_docs}\n\n"
        f"Conversation History:\n{formatted_history.strip()}\n\n"
        f"User Question:\n{user_question.strip()}"
    )