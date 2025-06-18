import os

import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS    # vector Store
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# load the GROQ and google API key from .env file

groq_api_key = os.getenv(groq_api_key)
os.environ['GOOGLE_API_KEY']=google_api_key

st.title("Q&A using Gemma")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

print(llm)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./ML notes")  # data Ingestion
        st.session_state.docs=st.session_state.loader.load()  # document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)



prompt1=st.text_input("What you want to ask from Notes?")

if st.button("Create Document Embeddings"):
    vector_embedding()
    st.write("Vector Store Database is Ready")
    num_vectors = st.session_state.vectors.index.ntotal
    st.write(f"Number of vectors in FAISS: {num_vectors}")


if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    response=retriever_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    # with streamlit expander
    with st.expander("Document Similarity Search"):
        # find the relevant chunks
        for i, doc in enumerate(response["context"]):  # using enumerate because interation with two variables
            st.write(doc.page_content)
            st.write("-----------------------------------")
