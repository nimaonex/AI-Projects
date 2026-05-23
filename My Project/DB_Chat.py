from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI 
import langchain_openai #assess
from langchain_community.tools import QuerySQLDataBaseTool
import pyodbc
import streamlit as st


conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=localhost;"
    "Database=AdventureWorksDW2019;"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
)

engine = create_engine(
    "mssql+pyodbc://",
    creator=lambda: conn
)

db = SQLDatabase(engine)

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="qwen2.5-3b-instruct",
    temperature=0
)

tools = [
    QuerySQLDataBaseTool(db=db)
]

prompt = """
     system,
     You are a professional data analyst.
     You are working with a SQL Server database.
     Rules:
      - Always use SQL queries
      - Join tables when needed
      - Never guess
      - If data is missing, say so
      - Answer clearly and concisely
"""
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix=prompt
)



st.title("💬 Chat with SQL Server")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
question = st.chat_input("Ask a question about the database")

if question:
    st.chat_message("user").write(question)
    result = agent.invoke({"input": question})
    st.chat_message("assistant").write(result)
    #st.write(result["output"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": result})