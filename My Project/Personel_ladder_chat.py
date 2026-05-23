
#load document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
#loader = PyPDFLoader("Personnel ladder.pdf")
loader = UnstructuredWordDocumentLoader("GS_En.docx")
document = loader.load()

#chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=80
)

docs = text_splitter.split_documents(document)

#embedding
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#vector store
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

#load model
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    #model="gemma-3-1b",
    model="qwen2.5-3b-instruct",
    temperature=0.3
)

#prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the provided context.\n\nContext:\n{context}"),
    ("human", "{question}")
])

prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        "تو یک دستیار تحلیل متن فارسی هستی. "
        "فقط و فقط بر اساس متن داده‌شده پاسخ بده. "
        "اگر پاسخ در متن وجود ندارد، دقیقاً بنویس: «در متن ذکر نشده است». "
        "از حدس زدن یا اضافه کردن اطلاعات خودداری کن. "
        "پاسخ را کوتاه، دقیق و به زبان فارسی بنویس.\n\n"
        "متن:\n{context}"
    ),
    ("human", "{question}")
])

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

#streamlit UI
import streamlit as st
st.title("📄 Chat with Your Document")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question about your document")

if user_input:
    st.chat_message("user").write(user_input)

    answer = rag_chain.invoke(user_input)

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})