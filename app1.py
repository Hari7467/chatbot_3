import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader   
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import time

from dotenv import load_dotenv
load_dotenv()

## load api key and Hf token
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
## define the embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## set up streamlit
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload the Pdf and chat with their content")

## Input the Groq Api Key
api_key=st.text_input("Enter your groq api key",type="password")

## check if groq api key is provided 
if api_key:
    llm=ChatGroq(api_key=api_key,model="gemma2-9b-it")
    ## Chat Interface
    
    session_id=st.text_input("Session ID",value="default_session")
    
    ## Statefully Manage The Chat History
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Choose A PDF File ",type="pdf",accept_multiple_files=True)
    ## process uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            
        ## split and create embeddings for the documents'
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        final_documents=text_splitter.split_documents(documents)
        vector=FAISS.from_documents(documents=final_documents,embedding=embeddings)
        retriever=vector.as_retriever()
        
        context_q_system_prompt=(
            "Given a chathostory and latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chathistory. Do not answer the question,"
            "just reformulate it if needed and otherwise return it as is."
        )
        context_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",context_q_system_prompt),MessagesPlaceholder("chathistory"),("human","{input}")
            ]
        )
        history_retriever=create_history_aware_retriever(llm,retriever,context_q_prompt)
        
        ## Answer Question
        
        ##prompt template
        system_prompt=(
            "you are an assistant for question and answer tasks:"
            "use the following pieces of rerieved context to answer."
            "if you don't know the anser for the question ,say that you don't know."
            "use three sentences maximum to provide answer and answer should be concise."
            "\n\n"
            "{context}"
        )
        qa_prompt=ChatPromptTemplate.from_messages([('system',system_prompt),MessagesPlaceholder("chathistory"),('human',"{input}")])
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_retriever,question_answer_chain)
        
        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversationaal_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )

        ## user input
        user_input=st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversationaal_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Plaese Enter the Groq Api Key")