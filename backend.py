""" TEMPORARY BACKEND SETUP FOR TESTING has nothing to do with app.py """


import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma


# Disable warnings
warnings.filterwarnings("ignore", category=Warning, module="langchain-community")


""" Deterministic partitioning """
def partition():
    # Loading the document
    file_path = "./content/painini.pdf"
    
    # Validating the document
    # Check file inside container
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist inside docker container")
    else:
        print(f"File exists at {os.path.abspath(file_path)}")
    # Check file size
    file_size = os.path.getsize(file_path)
    if (file_size < 1000):
        print("File is not a valid PDF")
    else:
        print(f"File size: {file_size}")
    
    loader = PyPDFLoader(file_path=file_path)
    document_string = loader.load()
    print(f"Document at {file_path} loaded successfully")
    
    # Configuring the text splitter
    text_splitter = RecursiveCharacterTextSplitter()
    text_splitter._chunk_size = 100 
    text_splitter._chunk_overlap = 1000
    
    # Splitting the text into chunks
    chunks = text_splitter.split_documents(document_string)
    print(f"Document has been split into {len(chunks)}")
    
    return chunks


""" Vectorization into embedding space"""
def vectorize():
    chunks = partition()
    
    # Configure the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Configure the vector db
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    print("Document vectorized succeessfully")
    
    return vector_db
    
""" RAG geometric indexing and retrieval """
def retrival(query):
    vector_db = vectorize()
    
    # Configuring the LLM provider
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    # Converts vector db into a callable function that vectorizes the input string 
    retriver = vector_db.as_retriever()
    
    # Defining the system prompt
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
        {"context": retriver, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Chaining successfull")
    
    # Invoke the chain operation
    return rag_chain.invoke(query)
    

""" The RAG backend of our app """
def run_backend():
    # The main user query
    query = "Who is painini?"
    
    print(f"Evaulating query vector:\n{query}")
    
    # Start the retrival chain
    response = retrival(query)
    
    print(f"Response:\n{response}")    
        
    
if __name__ == "__main__":
    run_backend()