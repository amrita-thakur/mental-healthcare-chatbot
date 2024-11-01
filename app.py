import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Mental HealthCare Chatbot"

# Load environment variables
load_dotenv(override=True)


def load_faiss_vector_store(db_directory_path: str = "faiss_db"):
    """
    Load FAISS vector store.

    Args:
        db_directory_path (str): path to FAISS database.
    Returns:
        FAISS vector store
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        vector_store = FAISS.load_local(db_directory_path, embeddings, allow_dangerous_deserialization=True)

        retriever=vector_store.as_retriever()

        return retriever
    
    except Exception as e:
        raise e

def load_chroma_vector_store(db_directory_path: str = "chroma_db"):
    """
    Load Chroma vector store.

    Args:
        db_directory_path (str): path to FAISS database
    Returns:
        FAISS vector store
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        vector_store = Chroma(persist_directory=db_directory_path, embedding_function=embeddings) 

        retriever=vector_store.as_retriever()

        return retriever
    
    except Exception as e:
        raise e
    
def create_conversational_chain(retriever: FAISS):
    """
    Create the conversational retrieval chain.

    Args:
        language_model (ChatGroq): The language model.
        retriever (FAISS): The vector database retriever.

    Returns:
        create_retrieval_chain: The conversational retrieval chain.
    """

    groq_api_key = st.secrets['GROQ_API_KEY']
    if not groq_api_key:
        raise ValueError("GROQ API key not found.")
    
    groq_model_name = "Llama3-8b-8192"
    language_model = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model_name)

    contextualize_q_system_prompt = "Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(language_model, retriever, contextualize_q_prompt)

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \

    {context}.
    Do not include, "According to the context" in the final output
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(language_model, qa_prompt)
    conversation_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  

    return conversation_chain

def get_session_history(store:dict, session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def initialize_session_state():
    """
    Initialize session state variables for chat history and messages.
    """
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

def handle_user_query(get_conversation_chain: create_retrieval_chain, user_query: str):
    """
    Handle the user query and get the response from the conversation chain.

    Args:
        conversational_rag_chain (create_retrieval_chain): The conversational retrieval chain.
        user_query (str): The user's query.

    Returns:
        str: The response from the conversation rag chain.
    """
    response = get_conversation_chain.invoke({"input": user_query, "chat_history": st.session_state['chat_history']})

    print(response)
    st.session_state['chat_history'].extend([HumanMessage(content=user_query), response["answer"]])

    return response["answer"]

def display_chat_interface(conversation_chain: create_retrieval_chain):
    """
    Display the chat interface using Streamlit.

    Args:
        conversational_rag_chain (create_retrieval_chain): The conversational retrieval chain.
    """
    st.title("Mental Healthcare Chatbot")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_query := st.chat_input("Ask about your Mental Health"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        response = handle_user_query(conversation_chain, user_query)

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """
    Main function to run the Streamlit application.
    """
    retriever = load_chroma_vector_store()
    conversation_chain = create_conversational_chain(retriever)

    initialize_session_state()
    display_chat_interface(conversation_chain)

if __name__ == "__main__":
    main()
