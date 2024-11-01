import os
import shutil
import pickle
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Mental HealthCare Chatbot"

# Load environment variables
load_dotenv(override=True)


def load_and_split_documents(directory_path: str = "data", chunk_size: int = 500, chunk_overlap: int = 0):
    """
    Load PDF documents from a directory and split them into chunks.

    Args:
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """

    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path not found: {directory_path}")
    
    pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap,
                                                    add_start_index=True,)
    document_chunks = chunk_splitter.split_documents(pdf_documents)

    return document_chunks

def save_to_chroma(document_chunks: list, db_path: str = "chroma_db"):
    """
    Initialize the language model and vector store.

    Args:
        document_chunks (list): List of document chunks.
        db_path (str): path to store data in chroma database locally.
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    Chroma.from_documents(document_chunks, embeddings, persist_directory=db_path)



if __name__ == "__main__":

    chunks = load_and_split_documents()
    save_to_chroma(chunks)
