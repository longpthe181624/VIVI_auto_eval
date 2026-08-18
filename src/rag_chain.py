import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Ensure project root and myenv site-packages are in sys.path
project_root = Path(__file__).resolve().parent.parent
site_packages = project_root / "myenv" / "lib" / "python3.14" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

import src.config as config


def get_vector_store() -> Chroma:
    """Loads the persistent Chroma vector store."""
    if not os.path.exists(config.CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"ChromaDB directory '{config.CHROMA_DB_DIR}' not found. "
            "Please run `python src/ingest.py` first to ingest your documents."
        )

    embeddings = config.get_embedding_model()

    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def format_docs(docs):
    """Formats retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGChatbot:
    """RAG Engine that handles context retrieval, prompt construction, and conversation history."""

    def __init__(self, top_k: int = 4):
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0.2,
            openai_api_key=config.OPENAI_API_KEY,
        )

        self.chat_history: List[BaseMessage] = []

        # System prompt instructions
        self.system_prompt = (
            "You are a helpful and knowledgeable RAG AI Assistant.\n"
            "Use the following retrieved context pieces to answer the user's question.\n"
            "If you do not know the answer based on the context, state clearly that "
            "the provided documents do not contain sufficient information.\n"
            "Keep your response structured, accurate, and concise.\n\n"
            "Context:\n{context}"
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answers a question using retrieved documents as context and returns sources."""
        # Retrieve relevant document chunks
        retrieved_docs = self.retriever.invoke(question)
        formatted_context = format_docs(retrieved_docs)

        # Build prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Chain execution
        chain = prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": formatted_context,
            "chat_history": self.chat_history,
            "question": question,
        })

        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))

        # Extract unique sources
        sources = list({
            doc.metadata.get("source", "Unknown Document") for doc in retrieved_docs
        })

        return {
            "answer": response,
            "sources": sources,
            "context_chunks": len(retrieved_docs),
        }

    def clear_history(self):
        """Clears the current conversation memory."""
        self.chat_history.clear()
