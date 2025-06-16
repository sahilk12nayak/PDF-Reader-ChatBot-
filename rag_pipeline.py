from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(persist_dir="db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )


    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain












# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA

# def build_rag_chain(persist_dir="db"):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#     llm = OpenAI(temperature=0.2)
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
#     return qa_chain
