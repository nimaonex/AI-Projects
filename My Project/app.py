import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Local LLM Chat", layout="centered")
st.title("💬 Local LLM Chat (LM Studio + Gemma)")

# ----------------------------
# Initialize LLM
# ----------------------------
@st.cache_resource
def load_model():
    return ChatOpenAI(
        model="gemma-3-1b",
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0.3,
    )

llm = load_model()

# ----------------------------
# Session state for chat
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# User input
# ----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Model response
    with st.chat_message("assistant"):
        response = llm.invoke(user_input)
        st.markdown(response.content)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )