import os
import sys
from pathlib import Path
from typing import List

# Ensure project root and myenv site-packages are in sys.path
project_root = Path(__file__).resolve().parent.parent
site_packages = project_root / "myenv" / "lib" / "python3.14" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
except ImportError:
    from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

import src.config as config


def load_documents(data_dir: str) -> List[Document]:
    """Loads all documents from the data directory (PDF, TXT, MD)."""
    data_path = Path(data_dir)
    if not data_path.exists():
        os.makedirs(data_path, exist_ok=True)
        print(f"Created data directory at {data_path}")
        return []

    documents = []

    # Load TXT and MD files
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
        show_progress=True,
    )
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
        show_progress=True,
    )
    documents.extend(md_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
        show_progress=True,
    )
    documents.extend(pdf_loader.load())

    return documents


def ingest_documents():
    """Main execution function to load, split, embed and store documents in ChromaDB."""
    print(f"Loading documents from: {config.DATA_DIR}")
    raw_docs = load_documents(config.DATA_DIR)

    if not raw_docs:
        print("No documents found in data directory. Add PDFs or text files to data/ and run again.")
        return None

    # Sanitize and validate loaded documents
    valid_docs = []
    for doc in raw_docs:
        if isinstance(doc, Document) and isinstance(doc.page_content, str) and doc.page_content.strip():
            valid_docs.append(doc)

    if not valid_docs:
        print("⚠️ No valid text content found in loaded documents.")
        return None

    print(f"Loaded {len(valid_docs)} valid document pages/files.")

    # Split documents into chunks safely
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(valid_docs)
    print(f"Split documents into {len(chunks)} chunks.")

    # Generate Embeddings & Persist Vector Database
    print("Generating embeddings and indexing into ChromaDB...")
    embeddings = config.get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )
    print(f"Successfully stored {len(chunks)} vectors in '{config.CHROMA_DB_DIR}'.")
    return vector_db


if __name__ == "__main__":
    ingest_documents()
