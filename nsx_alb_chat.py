import os
import streamlit as st
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

#Loading
def load_data():
    loader = PyPDFLoader(input_pdf)
    return loader.load()

#Indexing and storing
def embed_data_open_ai(_data):
    # OpenAI Embedding
    # Required Open AI key
    char_text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    doc_texts = char_text_splitter.split_documents(_data)
    embeddings = OpenAIEmbeddings()
    docs = Chroma.from_documents(doc_texts, embeddings)
    return docs

def embed_data_local_embedding(_data, local_embedding_path):
    # Automatically downloads the model if required
    # Flag is there for force download
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    print(f"Embedding: {embedding.e}")
    if os.path.exists(local_embedding_path):
        print(f"Loading FAISS index from {local_embedding_path}")
        vectorstore = FAISS.load_local(local_embedding_path, embedding, allow_dangerous_deserialization=True)
        print("done.")  
    else:
        print(f"Building FAISS index from documents")
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=75
                    )
        frags = text_splitter.split_documents(_data)
        print(f"Poplulating vector store with {len(_data)} docs in {len(frags)} fragments")
        vectorstore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {local_embedding_path}")
        # Save embedded docs locally
        vectorstore.save_local(local_embedding_path)
        print(f"Saved FAISS index to {local_embedding_path}")
    return vectorstore

def set_local_llm():
    return ChatOpenAI(
        openai_api_key="edhjwef",
        base_url="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=500,
        model='TheBloke/una-cybertron-7B-v2-GGUF'
    )

def set_openai_as_llm():
    return OpenAI(
        # temperature=0.7,
        # max_tokens=500,
        # model='gpt-3.5-turbo-0613'
    )

def conversational_retrieval(vstore):
    llm_mod = set_local_llm()
    # llm_mod = set_openai_as_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.load_memory_variables({})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_mod,
        memory=memory,
        retriever=vstore.as_retriever()
    )
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input=="exit":
            break
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain.invoke({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)    

def qa_retrieval(vstore):
    llm_mod = set_local_llm()
    # llm_mod = set_openai_as_llm()
    model = RetrievalQA.from_chain_type(llm=llm_mod, chain_type="stuff", retriever=vstore.as_retriever())
    #model = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vstore.as_retriever())
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input=="exit":
            break
        response = model.invoke(user_input)
        print("\n AI: " + response["result"])

def chat_app():
    data = load_data()
    # vstore = embed_data_open_ai(data)
    vstore = embed_data_local_embedding(data, "./local_embedding/")
    #render_demo(vstore)
    # qa_retrieval(vstore)
    conversational_retrieval(vstore)

parser = argparse.ArgumentParser()
parser.add_argument("--input_pdf_file")
args = parser.parse_args()
input_pdf = args.input_pdf_file
chat_app()
