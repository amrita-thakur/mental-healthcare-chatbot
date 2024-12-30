import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage


def initialize_session_state():
    """
    Initialize session state variables for chat history and messages.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def handle_user_query(get_conversation_chain: create_retrieval_chain, user_query: str):
    """
    Handle the user query and get the response from the conversation chain.

    Args:
        conversational_rag_chain (create_retrieval_chain): The conversational retrieval chain.
        user_query (str): The user's query.

    Returns:
        str: The response from the conversation rag chain.
    """
    response = get_conversation_chain.invoke(
        {"input": user_query, "chat_history": st.session_state["chat_history"]}
    )

    st.session_state["chat_history"].extend(
        [HumanMessage(content=user_query), response["answer"]]
    )

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
