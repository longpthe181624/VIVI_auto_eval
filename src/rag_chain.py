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


_CACHED_VECTOR_STORE = None

def get_vector_store(force_reload: bool = False) -> Chroma:
    """Loads the persistent Chroma vector store with global caching."""
    global _CACHED_VECTOR_STORE
    if _CACHED_VECTOR_STORE is not None and not force_reload:
        return _CACHED_VECTOR_STORE

    if not os.path.exists(config.CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"ChromaDB directory '{config.CHROMA_DB_DIR}' not found. "
            "Please run `python src/ingest.py` first to ingest your documents."
        )

    embeddings = config.get_embedding_model()

    _CACHED_VECTOR_STORE = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    return _CACHED_VECTOR_STORE


def reset_vector_store():
    """Clears the global cached vector store reference."""
    global _CACHED_VECTOR_STORE
    _CACHED_VECTOR_STORE = None


def format_docs(docs):
    """Formats retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGChatbot:
    """RAG Engine that handles context retrieval, prompt construction, and conversation history with offline fallbacks."""

    def __init__(self, top_k: int = 4):
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        self.chat_history: List[BaseMessage] = []

        self.system_prompt = (
            "You are a helpful and knowledgeable VinFast vehicle AI Assistant.\n"
            "Use the following retrieved context pieces to answer the user's question accurately.\n"
            "If you do not know the answer based on the context, state clearly that "
            "the provided documents do not contain sufficient information.\n"
            "Keep your response structured, accurate, and concise.\n\n"
            "Context:\n{context}"
        )

    def _get_llm(self):
        """Attempts to initialize OpenAI, HuggingFace Qwen2.5, or Ollama LLM."""
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
            try:
                return ChatOpenAI(
                    model=config.MODEL_NAME,
                    temperature=0.2,
                    openai_api_key=config.OPENAI_API_KEY,
                )
            except Exception:
                pass

        try:
            from transformers import pipeline
            from langchain_huggingface import HuggingFacePipeline
            hf_pipe = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                return_full_text=False,
            )
            return HuggingFacePipeline(pipeline=hf_pipe)
        except Exception as e:
            print(f"⚠️ HuggingFace LLM load note: {e}")

        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="qwen2.5:3b", timeout=5.0)
        except Exception:
            return None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answers a question using retrieved documents as context and returns sources."""
        retrieved_docs = self.retriever.invoke(question)
        sources = list(dict.fromkeys([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
        formatted_context = format_docs(retrieved_docs)

        llm = self._get_llm()
        if llm:
            try:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ])
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": formatted_context,
                    "chat_history": self.chat_history,
                    "question": question,
                })
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=response))
                return {"answer": response, "sources": sources}
            except Exception as e:
                print(f"⚠️ LLM invocation failed, using offline retrieval fallback: {e}")

        # Fallback Offline Knowledge Retrieval Engine
        if not retrieved_docs:
            answer = "Chưa tìm thấy thông tin phù hợp trong các tài liệu xe VinFast hiện có."
        else:
            answer = f"Thông tin tìm thấy từ tài liệu VinFast ({len(retrieved_docs)} trích đoạn liên quan):\n\n"
            for doc in retrieved_docs[:4]:
                src_name = doc.metadata.get("source", "Tài liệu VinFast")
                answer += f"📄 [{src_name}]:\n{doc.page_content.strip()}\n\n"

        return {"answer": answer, "sources": sources, "context_chunks": len(retrieved_docs)}

    def clear_history(self):
        """Clears the current conversation memory."""
        self.chat_history.clear()
