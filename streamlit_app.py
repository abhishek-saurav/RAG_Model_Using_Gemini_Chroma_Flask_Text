from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os
import streamlit as st 



# Initialize components
def initialize_qa_system():
    # Load documents
    loader = TextLoader('State_union.txt', encoding='utf-8')
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyA4Qh90opJDxEFrbfkZU6DuqozxawiyYDA',
        task_type="retrieval_query"
    )

    # Create the vector store
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

    # Define the prompt template
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

    # Set up safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # Set up the chat model with temperature
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key='AIzaSyA4Qh90opJDxEFrbfkZU6DuqozxawiyYDA',
        temperature=0.7,
        safety_settings=safety_settings
    )

    # Create the QA chain
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
    
    return qa_chain,vectordb

# Initialize QA chain globally
qa_chain,vectordb = initialize_qa_system()

# Streamlit app layout
st.title("Question Answering System")
st.write("Ask a question about the documents!")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Persistent input for user question
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# Define the main chatbot route
user_question = st.text_input("Your Question:", value=st.session_state.user_question)
if st.button("Get Answer"):
    if user_question:

        # Prepare the context (you can adjust this as needed)
        context_docs = vectordb.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(user_question)
        context_text = "\n".join([doc.page_content for doc in context_docs])  # Combine retrieved documents

        # Run the query through the QA chain
        response = qa_chain.invoke({"query": user_question, "context": context_text})

        # Extract the answer from the response
        # Assuming response has 'result' and 'source_documents' attributes
        bot_response = response['result']  # Get the generated answer
        st.write("Answer:", bot_response)

        # Update conversation history
        st.session_state.conversation_history.append({"user": user_question, "bot": bot_response})

        # Store the last response in session state
        st.session_state.last_response = response  # Save the response for later use

        # Update the user question in session state
        st.session_state.user_question = user_question  # Maintain user input
        
        # Save conversation history to a text file
        with open("conversation_history.txt", "a") as f:  # Append mode
            f.write(f"User: {user_question}\nBot: {bot_response}\n\n")


# Button to display source documents
if st.button("Show Source Documents"):
    # Check if last response exists in session state
    if 'last_response' in st.session_state:
        source_documents = st.session_state.last_response.get('source_documents', [])
        if source_documents:
            st.write("Source Documents:")
            for doc in source_documents:
                st.write(doc.page_content)  # Display source document content
        else:
            st.write("No source documents available. Please ask a question first.")
    else:
        st.write("Please get an answer first.")

    # Maintain the user question in the input field
    st.session_state.user_question = user_question  # Ensure it remains in the input field

    # Maintain the user question in the input field
    st.session_state.user_question = user_question  # Ensure it remains in the input field

# Button to display conversation history
if st.button("Show Conversation History"):
    # Read and display conversation history from the text file
    st.write("Conversation History:")
    with open("conversation_history.txt", "r") as f:
        history = f.readlines()
        for line in history:
            st.write(line.strip())  # Display each line of the conversation history

    # Maintain the user question in the input field
    st.session_state.user_question = user_question  # Ensure it remains in the input field
