from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import streamlit as st
import fitz  # PyMuPDF for PDF handling
import os  # For accessing environment variables

# Load the API key from the environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')  # Change your API key here

# Function to read PDF content
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Function to summarize text content using the same LLM
def summarize_text(text):
    summary_prompt = f"Please provide a concise summary of the following content in 3-4 lines:\n\n{text}\n\nSummary:"
    summary_response = qa_chain.invoke({"query": summary_prompt, "context": ""})  # Call your LLM to get the summary
    return summary_response['result']  # Return the generated summary

# Initialize components
def initialize_qa_system(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=API_KEY,  # Use global API key
        task_type="retrieval_query"
    )

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="path/to/persist/directory")

    prompt_template = """
    ## Safety and Respect Come First!

    You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

    * **Harassment or Bullying:** Targeting individuals or groups with hateful or hurtful language.
    * **Hate Speech:**  Content that attacks or demeans others based on race, ethnicity, religion, gender, sexual orientation, disability, or other protected characteristics.
    * **Violence or Harm:**  Promoting or glorifying violence, illegal activities, or dangerous behavior.
    * **Misinformation and Falsehoods:**  Spreading demonstrably false or misleading information.

    **How to Use You:**

    1. **Provide Context:** Give me background information on a topic.
    2. **Ask Your Question:** Clearly state your question related to the provided context.

    ##  Answering User Question:
    Context: \n {context}
    Question: \n {question}
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=API_KEY,  # Use global API key
        temperature=0.7,
        safety_settings=safety_settings
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, vectordb  # Return both qa_chain and vectordb

# Streamlit app layout
st.title("Question Answering System")
st.write("Upload a TXT or PDF file to ask questions about its content!")

# File uploader for user to upload text or PDF files
uploaded_file = st.file_uploader("Choose a TXT or PDF file", type=["txt", "pdf"])

# Check if a file has been uploaded
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        document_content = uploaded_file.read().decode("utf-8")
        documents = [Document(page_content=document_content)]
    elif uploaded_file.type == "application/pdf":
        document_content = read_pdf(uploaded_file)
        documents = [Document(page_content=document_content)]

    qa_chain, vectordb = initialize_qa_system(documents)

    # Display a quick summary of the document
    summary = summarize_text(document_content)  # Summarize the content
    st.write("Summary:", summary)  # Display the summary

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Persistent input for user question
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    # Define the main chatbot route
    user_question = st.text_input("Your Question:", value=st.session_state.user_question)

    if st.button("Get Answer"):
        if user_question:
            context_docs = vectordb.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(user_question)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            response = qa_chain.invoke({"query": user_question, "context": context_text})

            bot_response = response['result']
            st.write("Answer:", bot_response)

            # Update conversation history with memory optimization
            st.session_state.conversation_history.append({"user": user_question, "bot": bot_response})
            if len(st.session_state.conversation_history) > 10:  # Limit to 10 interactions
                st.session_state.conversation_history.pop(0)

            st.session_state.last_response = response
            st.session_state.user_question = user_question
            
            # Save conversation history to a text file
            with open("conversation_history.txt", "a") as f:
                f.write(f"User: {user_question}\nBot: {bot_response}\n\n")

    # Button to display source documents
    if st.button("Show Source Documents"):
        if 'last_response' in st.session_state:
            source_documents = st.session_state.last_response.get('source_documents', [])
            if source_documents:
                st.write("Source Documents:")
                for doc in source_documents:
                    st.write(doc.page_content)
            else:
                st.write("No source documents available. Please ask a question first.")
        else:
            st.write("Please get an answer first.")

    # Button to display conversation history
    if st.button("Show Conversation History"):
        st.write("Conversation History:")
        for exchange in st.session_state.conversation_history:
            st.write(f"User: {exchange['user']}")
            st.write(f"Bot: {exchange['bot']}")
else:
    st.warning("Please upload a text file or PDF to proceed.")