from urllib.parse import quote_plus
from datetime import datetime, UTC
from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session
)
from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    DateTime,
    create_engine
)
from sqlalchemy.exc import SQLAlchemyError
from utils import load_config
import os
from dotenv import load_dotenv



config = load_config()
load_dotenv(override=True)
###############################################################################################################
#LMStudio llm
# llm = ChatOpenAI(
#     base_url="http://localhost:1234/v1",
#     api_key="not-needed",
#     model="google/gemma-3-4b",
#     temperature=0,
#     max_completion_tokens=250
# )


llm = ChatOpenAI(
    base_url=config["LLM_URL"],
    api_key=os.getenv("API_KEY"),
    model=config["ai_model"],
    temperature=config["temperature"],
    max_completion_tokens=config["max_token"]
)
###############################################################################################################
#SQLAlchemy Connection

params = quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=AI;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    pool_pre_ping=True
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
###############################################################################################################
#Models
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "ChatMessages"

    id = Column(BigInteger, primary_key=True)
    sessionid = Column(String(100))
    role = Column(String(20))
    content = Column(Text)
    createdat = Column(DateTime)


class ConversationSummary(Base):
    __tablename__ = "ConversationSummary"

    sessionid = Column(String(100), primary_key=True)
    summary = Column(Text)
    updatedat = Column(DateTime)

###############################################################################################################
#Save Messages
def save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str
) -> None:
    
    msg = ChatMessage(
        sessionid=session_id,
        role=role,
        content=content,
        createdat = datetime.now(UTC)
    )

    db.add(msg)

###############################################################################################################
#Load Recent History
def load_recent_messages(
    db: Session,
    session_id: str,
    max_chars: int = 8000
):
    """
    Load the latest messages for a session and
    return them as LangChain message objects.
    
    """
    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.sessionid == session_id
        )
        .order_by(
            ChatMessage.createdat.desc()
        )
        .all()
    )

    total_chars = 0
    selected = []

    for row in rows:

        content_length = len(row.content)

        if total_chars + content_length > max_chars:
            break

        selected.append(row)
        total_chars += content_length
       
    rows.reverse()

    messages = []

    for row in rows:

        if row.role == "user":
            messages.append( f"{row.role}: {row.content}"
                # HumanMessage(
                #     content=row.content
                # )
            )

        elif row.role == "system":
            messages.append(f"{row.role}: {row.content}"
                # AIMessage(
                #     content=row.content
                # )
            )

    return "\n".join(messages)

###############################################################################################################
def load_summary(
    db: Session,
    session_id: str
) -> str:

    try:

        summary_record = (
            db.query(ConversationSummary)
            .filter(
                ConversationSummary.sessionid == session_id
            )
            .first()
        )

        if summary_record is None:
            return ""

        return summary_record.summary or ""

    except SQLAlchemyError:
        return ""
###############################################################################################################

def rewrite_question(history: str, question: str, summary: str) -> str:

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    Rewrite the user's latest question into a standalone question.

    Conversation Summary: {summary}

    Conversation History: {history}
    Instructions:
        - Use the summary and chat history to understand references.
        - Preserve the original meaning.
        - Replace pronouns such as "it", "they", "this", "that" with their actual meaning when possible.
        - Return only the rewritten question.
    """
            ),

            # MessagesPlaceholder(
            #     variable_name="chat_history"
            # ),

            ("human", "{question}")
        ]
    )

    rewrite_chain = (
        rewrite_prompt
        | llm
    )

    return rewrite_chain.invoke(
        {
            "history": history, #load_recent_history
            "question": question, 
            "summary": summary #load_summary
        }
    ).content

# next step: embed(rewrite_question)
# based on the function above, search qdrant -> build context -> answer prompt -> generating answer -> again save message

###############################################################################################################
def summarize_conversation(
    db: Session,
    session_id: str
):

    summary_prompt = ChatPromptTemplate.from_template(
    """
    summarize the latest user questions and system answers by incorporating
    current summary and the new lines of conversation below.
    so that the result reflects the most recent context and developments.

    Current Summary: {current_summary}

    Conversation: {conversation}

    Update the summary.

    Rules:
    - Keep important facts.
    - Keep user goals and topics.
    - Remove repetition.
    - Maximum 200 words.

    Updated Summary:
    """
    )

    summary_chain = (
        summary_prompt
        | llm
    )

    current_summary = load_summary(db=db, session_id=session_id)
    conversation = load_recent_messages(db=db, session_id=session_id)

    return summary_chain.invoke(
            {
                "current_summary": current_summary,
                "conversation": conversation
            }
        ).content

###############################################################################################################
def save_summary(
    db: Session,
    session_id: str,
    summary_text: str
):

    row = (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.sessionid == session_id
        )
        .first()
    )

    if row is None:

        row = ConversationSummary(
            sessionid=session_id,
            summary=summary_text,
            updatedat=datetime.now(UTC)
        )

        db.add(row)

    else:

        row.summary = summary_text
        row.updatedat = datetime.now(UTC)

    db.commit()
    db.close()




    ###############################################################################################################

# def get_messages_for_summary(
#     db: Session,
#     session_id: str,
#     max_messages: int = 20
# ) -> str:

#     rows = (
#         db.query(ChatMessage)
#         .filter(
#             ChatMessage.sessionid == session_id
#         )
#         .order_by(
#             ChatMessage.createdat.desc() #default was asc
#         )
#         .limit(max_messages)
#         .all()
#     )

#     lines = []

#     for row in rows:

#         lines.append(
#             f"{row.role}: {row.content}"
#         )

#     return "\n".join(lines)