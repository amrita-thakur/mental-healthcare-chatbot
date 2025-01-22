import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


def load_and_split_documents(directory_path: str = "data") -> List[str]:
    """
    Load PDF documents from a directory.

    Parameters:
        directory_path (int): folder path of data.

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
