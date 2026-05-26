from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from datetime import datetime, UTC
from LLM_Client import generate
from langchain_openai.chat_models import ChatOpenAI 

###############################################################################################################
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="google/gemma-3-4b",
    temperature=0,
    max_completion_tokens=250
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
###############################################################################################################
#Models
from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    DateTime
)

from sqlalchemy.orm import declarative_base

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
from sqlalchemy.orm import Session

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

#maybe needed in future for "Instead of manually creating sessions inside every endpoint"
# from fastapi import Depends
# @router.post("/ask")
# def ask(
#     req: AskRequest,
#     db: Session = Depends(get_db)
# ):

###############################################################################################################
#Load Recent History
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


def load_recent_history(
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
            messages.append(
                HumanMessage(
                    content=row.content
                )
            )

        elif row.role == "system":
            messages.append(
                AIMessage(
                    content=row.content
                )
            )

    return messages

###############################################################################################################

from sqlalchemy.exc import SQLAlchemyError

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


from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

def rewrite_question(history: str, question: str, summary: str) -> str:

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    Rewrite the user's latest question into a standalone question.

    Conversation Summary: {summary}

    Instructions:
        - Use the summary and chat history to understand references.
        - Preserve the original meaning.
        - Replace pronouns such as "it", "they", "this", "that" with their actual meaning when possible.
        - Return only the rewritten question.
    """
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            ("human", "{question}")
        ]
    )

    rewrite_chain = (
        rewrite_prompt
        | llm
    )

    return rewrite_chain.invoke(
        {
            "chat_history": history, #load_recent_history
            "question": question, 
            "summary": summary #load_summary
        }
    ).content

# next step: embed(rewrite_question)
# based on the function above, search qdrant -> build context -> answer prompt -> generating answer -> again save message
###############################################################################################################
#update summary every 20 messages
def should_update_summary(
    db: Session,
    session_id: str
):
    count = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.sessionid == session_id
        )
        .count()
    )

    return count % 20 == 0
###############################################################################################################

# def create_summary():

#     summary_instructions = f'''
#     update the ongoing summary by incorporating the new lines of conversation below.
#     build upon the pervious summary rather than repeating it so that the result reflects the most recent context
#     and developments

#     pervious summary:
#     {state.get("summary", "")}

#     '''
#     print(summary_instructions)

#     summary = llm.invoke([HumanMessage(summary_instructions)])

#     return State(messages= remove_messages, summary= summary.content)



from sqlalchemy.orm import Session

def get_messages_for_summary(
    db: Session,
    session_id: str,
    max_messages: int = 20
) -> str:

    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.sessionid == session_id
        )
        .order_by(
            ChatMessage.createdat.asc()
        )
        .limit(max_messages)
        .all()
    )

    lines = []

    for row in rows:

        lines.append(
            f"{row.role}: {row.content}"
        )

    return "\n".join(lines)


from langchain_core.prompts import ChatPromptTemplate

summary_prompt = ChatPromptTemplate.from_template(
"""
Current Summary:

{current_summary}

Conversation:

{conversation}

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


def save_summary(
    db: Session,
    session_id: str,
    summary_text: str
):

    row = (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.session_id == session_id
        )
        .first()
    )

    if row is None:

        row = ConversationSummary(
            session_id=session_id,
            summary=summary_text,
            updated_at=datetime.now(UTC)
        )

        db.add(row)

    else:

        row.summary = summary_text
        row.updated_at = datetime.now(UTC)

    db.commit()