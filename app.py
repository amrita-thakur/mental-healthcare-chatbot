from dotenv import load_dotenv

from db.db_helper import load_faiss_vector_store
from llm.langchain_utils import create_conversational_chain
from app_utils.streamlit_utils import initialize_session_state, display_chat_interface

load_dotenv(override=True)


def main():
    """
    Main function to run the Streamlit application.
    """
    retriever = load_faiss_vector_store()
    conversation_chain = create_conversational_chain(retriever)

    initialize_session_state()
    display_chat_interface(conversation_chain)


if __name__ == "__main__":
    main()
