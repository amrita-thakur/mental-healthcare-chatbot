import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def save_to_faiss(docs_texts: list, db_path: str = "faiss_db_raptor"):
    """
    Initialize the language model and vector store.

    Parameters:
        document_chunks (list): List of document chunks.
        db_path (str): path to store data in FAISS database locally.
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(db_path):
        shutil.rmtre

    # create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
    )
    faiss_vector_database = FAISS.from_texts(texts=docs_texts, embedding=embeddings)

    faiss_vector_database.save_local(db_path)


def load_faiss_vector_store(db_directory_path: str = "faiss_db_raptor"):
    """
    Load FAISS vector store.

    Parameters:
        db_directory_path (str): path to FAISS database.
    Returns:
        FAISS vector store
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
        )
        vector_store = FAISS.load_local(
            db_directory_path, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever()

        return retriever

    except Exception as e:
        raise e
