import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# UI configuration
st.set_page_config(page_title="Chat With PDF", layout="wide")
st.title("Document Semantic Search Engine")


# Heap memory allocation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None


# I/O processig pipeline
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    # If the file and the embeddings are not cached inside the server ram (heap) (cache miss) execute this to prevent expensive calls repeatedly
    if uploaded_file is not None and st.session_state["vector_db"] is None:
        with st.spinner("Executing, chunking and vector projection..."):
            
            # Write the file payload that is coming through the websocket to a temporary file inside the server os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filepath = temp_file.name
                
            # Deterministic partitioning
            loader = PyPDFLoader(temp_filepath)
            documents = loader.load()
            # Splitter configuration 
            splitter = RecursiveCharacterTextSplitter()
            splitter._chunk_size = 1000
            splitter._chunk_overlap = 100
            chunks = splitter.split_documents(documents)
            
            # Vector projection
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
            vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            
            # Cache the memory pointer
            st.session_state["vector_db"] = vector_db
            st.success("Vectorization complete. Vector DB cached.")
            
            
# UI rendering from state
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        

# Network payload listener (chat input)
user_query = st.chat_input("Input you query...")

if user_query:
    # Halt execution if the the vector db is not cached
    if st.session_state["vector_db"] is None:
        st.error("Null Pointer Exception. A document must be vectorized before querying it.")
        st.stop()
        
    # Append user string to the heap render
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    
    # LCEL execution
    with st.chat_message("assistant"):
        with st.spinner("answering..."):
            # Configuring the LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=1.0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            
            # Fetch the vector db via it's pointer 
            retriever = st.session_state["vector_db"].as_retriever()
            
            system_prompt = (
                "You are an helpful assistant for question-answering tasks."
                "Use the following pices of retrieved context to answer the question."
                "If you don't know the answer, output exactly: 'Insufficient data'."
                "\n\n"
                "context: {context}"
            )
            
            # Prompt setup
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}")
            ])
            
            # Chaining
            rag_chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            print("Chaining successfull")
    
            # Invoke the chain operation
            response_string = rag_chain.invoke(user_query)
            st.markdown(response_string)
    
    # Append final system string to heap
    st.session_state["chat_history"].append({"role": "assistant", "content": response_string})