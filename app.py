import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "put_your_api_key_here"  # Fallback to the provided key
genai.configure(api_key=api_key)

#read each page of pdf and extract all text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#break text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#vector embedding of (pdf) data
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    


#identify context and answer
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Use the correct model name for Gemini - updated to the current available model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                                  temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Using deprecated chain, but keeping for compatibility
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)#get context (retrieved)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question}, #context + que ->LLM->
            return_only_outputs=True)
        
        return response["output_text"]#return answer from LLM
        # return st.write("user", response["output_text"])#*#return answer from LLM
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        print(f"Error details: {e}")

#main
def main():
    st.set_page_config(page_title="Chat PDF")

    # # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

    # st.header("Chat with PDF using Gemini💁")
    st.title("Help_Buddy: A virtual Assistant")
    st.subheader("Welcome 💁 to Aliah University student helpdesk")
    #display chat history from history on app rerun
    for chat_history in st.session_state.chat_history:
        with st.chat_message(chat_history["role"]):
            st.markdown(chat_history["content"])

    # Input area
    # user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question := st.chat_input("Ask your Question related to.. "):
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get answer from the model

    if user_question:
        answer=user_input(user_question)
        with st.chat_message("assistant"):
           st.session_state.chat_history.append({"role": "assistant", "content": answer})  # Add answer to chat history
           st.markdown(answer)  # Display the answer in the chat message
        
           

# Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Content:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
                        print(f"Error details: {e}")

        footer_html = """
        <style>
        .copyright{   
            position: fixed;
            bottom: 8px;
            # right: 10px;
            left :10px;
            font-size: 12px;
            color: light-dark(#333, #fff);
            font-weight: bold;
        }
        </style>
          <div class='copyright'>
                  <p>&copy; 2025 Shahzad Alam</p>
                  <p>Developed in ❤️ with Aliah University.</p>
                    </div>
"""
        st.markdown(footer_html, unsafe_allow_html=True)          
 

                  
    
if __name__ == "__main__":
    main()