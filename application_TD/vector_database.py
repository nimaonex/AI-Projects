from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from dataframe import DataFrame
from utils import load_config
from sentence_transformers import SentenceTransformer


config = load_config()

if config.get("embedding_method") == "SBERT":
        try:
            model = SentenceTransformer(config["embedding_model"])
        except Exception as e:
            raise RuntimeError(f"Failed to load SBERT model: {e}")

def load_vector_db(
        db_type: str = config["db_type"],
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
        embedding_method: str = config["embedding_method"],  # New parameter to select embedding method
        vectorizer = None
):
    """
    Load a vector database with the specified parameters.

    This function initializes an embedding model and loads a vector database (Chroma by default)
    using the specified parameters such as database type, embedding model, chunk size, and chunk overlap.

    Args:
        company_name (str): The name of the company for which the vector database is being loaded.
        db_type (str): The type of vector database to load. Default is "chroma".
        chunk_size (int): The maximum size of each chunk in characters. Default is 1600.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 200.
        embedding_method (str): The method to use for creating embeddings ("SBERT" or "tfidf").

    Returns:
        The loaded vector database.
    """

    embedding_model: str = config["embedding_model"]

    if db_type == "chroma":
        # Determine the persist directory based on embedding model, chunk size, and chunk overlap
        persist_directory = f"Database/chroma_db/{embedding_model}/{chunk_size}_{chunk_overlap}"
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Load the Chroma vector database
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        raise ValueError(f"Unsupported db_type: {db_type} for {embedding_method} embedding method!")

    return db

#########################################################################################################################


def initialize_similarity_search(
        #company_id: int,
        db: DataFrame,
        query: str,
        is_private: int,
        top_k_documents: int = config["top_k_documents"],
        score_threshold: float = config["score_threshold"],
        mmr: bool = config["maximum_marginal_relevance"],
        db_type: str = config["db_type"],  # New parameter to specify the database type
        embedding_method: str = config["embedding_method"]
) -> Tuple[List[str], List[str], List[int], list[int]]:
    """
    Initialize and return the similarity search chain.

    This function performs a similarity search on the given database using the provided query.
    It supports both standard similarity search and Maximal Marginal Relevance (MMR) search.

    Args:
        db: The vector database to search.
        query (str): The search query string.
        top_k_documents (int): The number of top documents to retrieve.
        score_threshold (float): The score threshold for filtering documents.
        mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for the search. Default is False.
        db_type (str): The type of the vector database ("chroma" or "mongodb"). Default is "chroma".

    Returns:
        Tuple[List[str], List[str], List[int]]: A tuple containing three lists:
            - prob_docs: List of retrieved document contents.
            - sources: List of document sources.
            - scores: List of document relevance scores.
            - query_vector: embedded query of the user
    """

    if db_type == "chroma":
        if mmr:
            # Perform Maximal Marginal Relevance (MMR) search
            docs = db.max_marginal_relevance_search(
                query=query,
                k=top_k_documents,
                score_threshold=score_threshold,
            )
            prob_docs, sources, scores = [], [], []
            for doc in docs:
                prob_docs.append(doc.page_content)
                #sources.append((doc.metadata["source"], doc.metadata["path"])) TODO: Remove If Returning Link Address Is Canceled
                sources.append(doc.metadata["source"])
                scores.append(0)  # Placeholder score, adjust if actual scores are available
        else:
            # Perform standard similarity search with relevance scores
            docs_and_scores = db.similarity_search_with_relevance_scores(
                query=query,
                k=top_k_documents,
                score_threshold=score_threshold,
            )
            prob_docs, sources, scores = [], [], []
            for doc, score in docs_and_scores:
                prob_docs.append(doc.page_content)
                sources.append(doc.metadata["source"])
                scores.append(int(round(score, 2) * 100))  # Convert score to percentage and round
    else:
        raise ValueError(f"Unsupported db_type: {db_type}")

    return prob_docs, sources, scores