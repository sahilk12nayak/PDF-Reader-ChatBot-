from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def ingest_documents(doc_dir="docs", persist_dir="db"):
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"Document directory '{doc_dir}' not found. Please create it and add PDFs.")

    loader = DirectoryLoader(doc_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("Documents successfully ingested into vector store.")

if __name__ == "__main__":
    ingest_documents()





