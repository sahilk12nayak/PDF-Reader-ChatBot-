import streamlit as st
from langchain_pipeline import build_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG-powered Chatbot")
st.markdown("Ask questions based on your custom documents.")

qa_chain = build_rag_chain()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": query})
        st.markdown("### Answer")
        st.success(result["result"])

        with st.expander("Source Documents"):
            for i, doc in enumerate(result.get("source_documents", [])):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}")
                st.code(doc.page_content[:500], language="markdown")
