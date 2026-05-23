#Indexing
#Cosine similarity, find similarity when embedding
#cos@ = ax*bx + ay*by , total similarity means @ = 0, cos0 = 1 else it will decrease. cos45 = 0.7, cos60 = 0.5
#it is just a 2 dimension example. in real world it is a multi-dimension model.
#Openai recommended model: "text-embedding-3" samll version(1536 D) , large version(3072 D)

#Retrieving
#retrieve most relevanet but also diverse chunks of data

#Augmented Generation
#user input and retrieved text are combined and fed into LLM model to generate contextually aligned response.

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.vectorstores import Chroma
from langchain_core.documents import Document
import copy
import numpy as np

document_splitter = CharacterTextSplitter(separator = ".",
                                          chunk_size = 500,
                                          chunk_overlap = 50)

#pdf
pdf_file = PyPDFLoader(f"E:\AI Projects\AI Course\AI_Files\Introduction_to_Data_and_Data_Science.pdf")
pdf_load = pdf_file.load()
#avoid all new line characters \n to decrease API cost
pdf_cut = copy.deepcopy(pdf_load) #to avoid over-riding the original file
for i in pdf_cut:
    i.page_content = ' '.join(i.page_content.split())
pdf_cut_split = document_splitter.split_documents(pdf_cut)

#docx
#docx_file = Docx2txtLoader(f"Files\English_leadership.docx")
#docx_load = docx_file.load()
#consider a document that have a headers and we mark them with # ## ### , ...
#we want to 
#markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#','Course Title'),
#                                                    ('##','Lecture Title')])
#pages_md_split = markdown_splitter.split_text(docx_load[0].page_content)

#for i in range(len(pages_md_split)):
#    pages_md_split[i].page_content = ' '.join(pages_md_split[i].page_content.split())

#docx_load_splitter = document_splitter.split_documents(pages_md_split)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vector1 = embedding.embed_query(pdf_cut_split[3].page_content)
vector2 = embedding.embed_query(pdf_cut_split[5].page_content)
vector3 = embedding.embed_query(pdf_cut_split[18].page_content)

np.dot(vector1,vector2), np.dot(vector1,vector3), np.dot(vector2,vector3)
#to ensure the vectors created with embedding function have a magnitude of 1
np.linalg.norm(vector1), np.linalg.norm(vector2), np.linalg.norm(vector3)

vectorstore = Chroma.from_documents(documents=pdf_cut_split,
                                    embedding=embedding,
                                    persist_directory=f"E:\AI Projects\AI Course\VS") 
#directory it is not necessary, but if you don't consider it, it remain until the kernel is restarted or shutdown
#we can use saved directory as below
Chroma(persist_directory=f"E:\AI Projects\AI Course\VS", embedding_function=embedding)

vectorstore.get() 
len(vectorstore.get()['documents'])
#the result is a dictionary with a list of ids in it as a Guid, to show the vectors for each Guid use code below
vectorstore.get(ids="selected Guid",
                include="embeddings")

#add document to the data base
added_document = Document(page_content = 'added document',
                          metadata = {
                              'Course Title':'Introduction to AI',
                              'Lecture Title':'RAG'
                          })
vectorstore.add_documents([added_document])

#update the document
updated_document = Document(page_content = 'updated document',
                            metadata = {
                                'Course Title':'Introduction to AI',
                                'Lecture Title':'RAG'
                          })

vectorstore.update_document(document_id = "select a Guid from document we want to be updates",
                            document = updated_document)

#delete
vectorstore.delete("Guid")

vectorstore.get("Guid")

#Retrieving
question = "what programming language do data scientist use?"
retrieved_document = vectorstore.similarity_search(query=question,
                                                   k = 5) #default is 4, it means number of document retrieved
#for better result
for i in retrieved_document:
    print(f"page_content: {i.page_content}\n--------\nLecture Title:{i.metadata['Lecture Title']}\n")
#the code above have some disadvantages. maybe one of those 5 retrieved documents, repeated and it is not perfect

#we can solve it by Maximal Marginal Relevance (MMR) search algorithm
#relevance means, calculating similarity and max similarity: MMR = similarity - max(similarity)
#lamda*similarity - (1-lambda)*max(similarity). lambda is somthing between 0 - 1 and it is controlling the balance
#between relevance and diversity. lambda 1 means more similarity and lambda 0 means more diversity
retrieved_document = vectorstore.max_marginal_relevance_search(query=question,
                                                               k=3,
                                                               lambda_mult=0.7,
                                                               filter={"Lecture Title":
                                                                       "Programming language & software Employed in Data Science - ..."}) #for better result
for i in retrieved_document:
    print(f"page_content: {i.page_content}\n--------\nLecture Title:{i.metadata['Lecture Title']}\n")

#other ways of improvement
retriever = vectorstore.as_retriever(search_type = 'mmr',
                                     search_kwargs = {'k':3, 
                                                      'lambda_mult': 0.7})
#the code above is a Runnable
retrieved_runnable = retriever.invoke(question)

#Generation
#join chain and RAG docs