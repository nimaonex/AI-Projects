import pandas as pd 
df_sale = pd.read_excel(f"Files/Sales_data.xlsx", sheet_name="Sales")
df_customer = pd.read_excel(f"Files/Sales_data.xlsx", sheet_name="Customer")
df_product = pd.read_excel(f"Files/Sales_data.xlsx", sheet_name="Product")

data_frames = {
    "Sales" : df_sale,
    "Customer" : df_customer,
    "Product" : df_product
}

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",   # LM Studio
    api_key="not-needed",
    model="qwen2.5-3b-instruct",
    temperature=0
)


from langchain_core.prompts import ChatPromptTemplate
Prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional data analyst.\n"
     "You are working with a pandas DataFrame called `df`.\n"
     "Rules:\n"
     "- Always use Python pandas code to answer.\n"
     "- Do NOT guess.\n"
     "- If data is missing, say so clearly.\n"
     "- Be concise and accurate.\n"
     "- Answer in English.\n"
    ),
    ("human", "{input}"),
])

RELATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional data analyst.\n\n"

     "You have access to the following pandas DataFrames:\n"
     "- customer\n"
     "- sales\n"
     "- product\n\n"

     "Relationships:\n"
     "- customer.customerid = sales.customerid\n"
     "- product.productid = sales.productid\n\n"

     "Rules:\n"
     "- Always use pandas code\n"
     "- Use df.merge() to join tables when needed\n"
     "- Never guess\n"
     "- Answer accurately and concisely\n"
     "- Answer in English\n"
    ),
    ("human", "{input}")
])

from langchain_experimental.agents import create_pandas_dataframe_agent
agent = create_pandas_dataframe_agent(
    llm,
    df = data_frames,
    verbose=True,
    prompt = Prompt,
    allow_dangerous_code=True  # required for analysis
)


import streamlit as st
# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- CHAT INPUT ----------
user_input = st.chat_input("Ask a question about the Excel data")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            result = agent.invoke(user_input)
            response = result["output"]
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )