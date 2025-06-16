import streamlit as st
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="RAG-powered Chatbot", layout="wide")

# Initialize chain only once
@st.cache_resource
def get_qa_chain():
    return build_rag_chain()

qa_chain = get_qa_chain()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("RAG-powered Chatbot")
st.markdown("Ask questions based on your custom documents.")

# Safely load the RAG chain
try:
    qa_chain = build_rag_chain()
except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()

# UI input box
query = st.text_input("Ask a question:")

# If user has entered a question
if query:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain({"query": query})
            st.markdown("### Answer")
            st.success(result["result"])

            # Show source documents
            with st.expander("Source Documents"):
                for i, doc in enumerate(result.get("source_documents", [])):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}")
                    st.code(doc.page_content[:500], language="markdown")

        except Exception as e:
            st.error(f"Failed to get a response: {e}")










# import streamlit as st
# from rag_pipeline import build_rag_chain

# st.set_page_config(page_title="RAG Chatbot", layout="wide")

# st.title("RAG-powered Chatbot")
# st.markdown("Ask questions based on your custom documents.")

# qa_chain = build_rag_chain()

# query = st.text_input("Enter your question:")
# if query:
#     with st.spinner("Generating answer..."):
#         result = qa_chain({"query": query})
#         st.markdown("### Answer")
#         st.success(result["result"])

#         with st.expander("Source Documents"):
#             for i, doc in enumerate(result.get("source_documents", [])):
#                 st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'N/A')}")
#                 st.code(doc.page_content[:500], language="markdown")
