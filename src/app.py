# Import libraries
import os
import re
import pandas as pd
import streamlit as st
from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from nltk.corpus import stopwords
import nltk

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load the API key
groq_api_key = os.getenv("groq_api_key")

# Streamlit Page Config
st.set_page_config(page_title="Medical Chatbot", page_icon="⚕️", layout="centered")
st.title("⚕️ Medical Chatbot")

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    train_data = dataset['train']
    df = pd.DataFrame(train_data[:100])  # Use first 100 rows
    df = df[["Patient", "Doctor"]].rename(columns={"Patient": "question", "Doctor": "answer"})

    # Download stopwords if needed
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()
        text = text.lstrip('q')  # Remove leading 'q'
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        words = text.split()
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return " ".join(words)

    # Preprocess questions and answers
    df['question'] = df['question'].astype(str).apply(preprocess_text)
    df['answer'] = df['answer'].astype(str).apply(preprocess_text)
    return df

df = load_and_preprocess_data()

# Combine data into a single dataset
documents = (df['question'] + ' ' + df['answer']).tolist()

# Create the vector store
@st.cache_resource
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(documents, embeddings)

vector_store = create_vector_store(documents)

# Create a retriever
retriever = vector_store.as_retriever()

@st.cache_resource
def load_llm_with_groq(api_key):
    return ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
# Load the LLM
llm = load_llm_with_groq(groq_api_key)


# Define a prompt template
prompt= ChatPromptTemplate.from_template(
    """
    You're a health assistant. Please abide by these guidelines:
    - Keep your sentences short, concise, and easy to understand.
    - Be concise and relevant: Most of your responses should be a sentence or two, unless you’re asked to go deeper.
    - If you don't know the answer, just say that you don't know; don't try to make up an answer.
    - Use three sentences maximum and keep the answer as concise as possible.
    - Always say "Thanks for asking!" at the end of the answer.
    - Use the following pieces of context to answer the question at the end:
    {context}
    Question: {input}
    """
)

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain=create_stuff_documents_chain(llm,prompt)

from langchain.chains import create_retrieval_chain
chain=create_retrieval_chain(retriever,document_chain)



# Streamlit app for interaction
user_input = st.text_input("Ask the Doctor..")

if user_input:
    response = chain.invoke({"input": user_input})
    st.write("Answer:", response["answer"])