import os
import streamlit as st

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_embeddings(provider: str = "huggingface"):
    """
    Retrieve embeddings model based on the specified provider.

    Args:
        provider (str): The provider for the embeddings. Defaults to 'huggingface'.

    Returns:
        Embeddings model instance.
    """
    if provider.lower() == "huggingface":
        try:
            model_name = st.secrets("EMBEDDING_MODEL_NAME")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize HuggingFace embeddings: {e}")
    else:
        raise ValueError(f"Unsupported provider for embeddings: {provider}")

    return embeddings


def get_llm(provider: str = "azure"):
    """
    Retrieve an LLM model based on the specified provider.

    Args:
        provider (str): The provider for the LLM. Defaults to 'azure'.

    Returns:
        LLM model instance.
    """
    if provider.lower() == "azure":
        try:
            model = AzureChatOpenAI(
                openai_api_key=st.secrets("AZURE_OAI_KEY"),
                azure_endpoint=st.secrets("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=st.secrets("AZURE_OPENAI_DEPLOYMENT"),
                openai_api_version=st.secrets("AZURE_OPENAI_API_VERSION"),
                openai_api_type="openai",
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Azure OpenAI model: {e}")

    elif provider.lower() == "groq":
        try:
            model = ChatGroq(
                groq_api_key=st.secrets("GROQ_API_KEY"),
                model_name=st.secrets("GROQ_MODEL_NAME", "default-groq-model"),
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq model: {e}")
    else:
        raise ValueError(f"Unsupported provider for LLM: {provider}")

    return model


def create_conversational_chain(retriever: FAISS):
    """
    Create the conversational retrieval chain.

    Args:
        retriever (FAISS): The vector database retriever.

    Returns:
        create_retrieval_chain: The conversational retrieval chain.
    """
    language_model = get_llm(provider="groq")

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

    history_aware_retriever = create_history_aware_retriever(
        language_model, retriever, contextualize_q_prompt
    )

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
    conversation_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return conversation_chain
