import os
import requests
import streamlit as st # type: ignore
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import anthropic 
import voyageai

def load_voyage_api_key():
    load_dotenv()
    return os.getenv("VOYAGE_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings_from_voyage(api_key, text_chunk):
    client = voyageai.Client(api_key=api_key)
    response = client.embed([text_chunk], model="voyage-2", input_type="document")
    return response.embeddings[0]

def get_vectorstore(text_chunks):
    api_key = load_voyage_api_key()
    embeddings = []
    for chunk in text_chunks:
        embedding = get_embeddings_from_voyage(api_key, chunk)
        embeddings.append(embedding)
    # Implement vector store logic here, e.g., saving embeddings to a database or in-memory storage
    return embeddings

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConverversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                           
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
    st.session_state.conversation

if __name__ == '__main__':
    main()