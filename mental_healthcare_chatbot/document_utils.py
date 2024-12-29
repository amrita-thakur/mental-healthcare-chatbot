import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


def load_and_split_documents(directory_path: str = "data") -> List[str]:
    """
    Load PDF documents from a directory and split them into chunks.

    Parameters:
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """

    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path not found: {directory_path}")

    pdf_loader = DirectoryLoader(
        directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    docs = pdf_loader.load()

    docs_texts = [d.page_content for d in docs]

    return docs_texts
