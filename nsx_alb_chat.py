import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

#Loading
def load_data():
	loader = PyPDFLoader("guides/Cloud-Console-Guide.pdf")
	return loader.load()

#Indexing and storing
def store_data(_data):
	char_text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
	doc_texts = char_text_splitter.split_documents(_data)
	embeddings = OpenAIEmbeddings()
	docs = Chroma.from_documents(doc_texts, embeddings)
	return docs

#query
def process_query(vstore, query):
	model = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vstore.as_retriever())
	return model.invoke(query)

def render_demo(vstore):
	st.title("NSX-ALB Cloud Console Chat Bot Demo")
	reader = st.text_input("Enter Query")
	if st.button("Get Answer"):
		st.write(process_query(vstore, reader))

def render_console(vstore):
	print("\n\t\t\t\tNSX-ALB Cloud Console Chat Bot Demo")
	while True:
		question = input("\n\n Please enter your question:  ")
		print("\n Answer: " + process_query(vstore, question)["result"].strip())
		quite = input("\n\nPlease enter y/n for next question: ")
		if quite.lower() != 'y':
			break
	return

def chat_app():
	data = load_data()
	vstore = store_data(data)
	#render_demo(vstore)
	render_console(vstore)

chat_app()
