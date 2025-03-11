import os
import io
import csv
import requests
import pandas as pd
import streamlit as st
from io import StringIO
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Free provider
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import List, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ------------------------------------------------------------------------------
# Function to load CSV from URL and convert it to a list of dictionaries
def csv_from_url_to_dict(url: str) -> List[dict]:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching the URL: {url}")
    csv_content = response.content.decode('utf-8')
    csv_file = io.StringIO(csv_content)
    data_dict = []
    csv_reader = csv.DictReader(csv_file)
    keys = csv_reader.fieldnames[0].split(';')
    for row in csv_reader:
        dictionary = {}
        values = list(row.values())[0].split(';')
        for i in range(len(values)):
            dictionary[keys[i]] = values[i]
        data_dict.append(dictionary)
    return data_dict

# Helper to convert a list of dictionaries to Document objects
def docs_from_dicts(dict_list: List[dict]) -> List[Document]:
    docs = []
    for item in dict_list:
        content = "\n".join([f"{k}: {v}" for k, v in item.items()])
        docs.append(Document(page_content=content))
    return docs

# ------------------------------------------------------------------------------
# File loading functions for manual uploads
def load_csv_file(file_obj) -> List[Document]:
    content = file_obj.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    docs = []
    for _, row in df.iterrows():
        text = row.to_string()
        docs.append(Document(page_content=text))
    return docs

def load_text_file(file_obj) -> Document:
    filename = file_obj.name.lower()
    if filename.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(file_obj)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            text = f"Error reading PDF: {e}"
    else:
        text = file_obj.getvalue().decode("utf-8")
    return Document(page_content=text)

# ------------------------------------------------------------------------------
# Prepare documents from uploads or URL
def prepare_documents(csv_upload, csv_url, privacy_upload, terms_upload) -> List[Document]:
    docs = []
    if csv_url.strip():
        try:
            dict_list = csv_from_url_to_dict(csv_url.strip())
            docs.extend(docs_from_dicts(dict_list))
        except Exception as e:
            st.error(f"Error loading CSV from URL: {e}")
    elif csv_upload is not None:
        docs.extend(load_csv_file(csv_upload))
    
    if privacy_upload is not None:
        docs.append(load_text_file(privacy_upload))
    if terms_upload is not None:
        docs.append(load_text_file(terms_upload))
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    split_docs = []
    for doc in docs:
        split_docs.extend(splitter.split_text(doc.page_content))
    return [Document(page_content=t) for t in split_docs]

def create_vector_store(docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ------------------------------------------------------------------------------
# Custom prompt template to restrict chatbot responses and include conversation history
# Modified prompt template without history (for initial working version)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a consumer chatbot for our smartphone business. You have access only to the provided smartphone data, privacy policy, and terms & conditions.\n"
        "Only use the provided context to answer questions. If a user asks a question outside these topics, reply: \"I'm sorry, I can only answer queries regarding our products, privacy policy, and terms & conditions.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

# ------------------------------------------------------------------------------
# Google Gemini LLM wrapper using the actual API call
class GoogleGemini(LLM):
    api_key: str = ""

    class Config:
        extra = "forbid"
        allow_mutation = True

    def __init__(self, api_key: str, **kwargs):
        # Initialize the parent class (LLM)
        super().__init__(**kwargs)
        # Set the api_key attribute using object.__setattr__ to bypass immutability
        object.__setattr__(self, "api_key", api_key)
        # Mark the field as set in the pydantic model
        self.__pydantic_fields_set__ = {"api_key"}

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            json_response = response.json()
            try:
                result = json_response["candidates"][0]["content"]["parts"][0]["text"]
                return result
            except Exception as e:
                return f"Error parsing response: {e}"
        else:
            return f"API error: {response.status_code} - {response.text}"

    def predict(self, prompt: str, **kwargs) -> str:
        return self._call(prompt, kwargs.get("stop"))


# ------------------------------------------------------------------------------
# Build the RetrievalQA chain using Google Gemini LLM with custom prompt
def build_qa_chain(api_key: str, vector_store) -> ConversationalRetrievalChain:
    gemini_llm = GoogleGemini(api_key=api_key)
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=gemini_llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )


# ------------------------------------------------------------------------------
# Main Streamlit app
def main():
    st.title("Smartphone Chatbot with Google Gemini API")
    
    st.write("### Provide your CSV data (upload or URL):")
    csv_upload = st.file_uploader("Upload Smartphone CSV", type=["csv"])
    csv_url = st.text_input("Or enter CSV URL:", "")
    
    st.write("### Upload additional files:")
    privacy_upload = st.file_uploader("Upload Privacy Policy (PDF or TXT)", type=["txt", "pdf"], key="privacy")
    terms_upload = st.file_uploader("Upload Terms & Conditions (PDF or TXT)", type=["txt", "pdf"], key="terms")
    
    st.write("### Enter your Google Gemini API Key:")
    api_key_input = st.text_input("API Key", "", type="password")
    
    st.write("### Ask a question:")
    user_query = st.text_input("Your Question:", placeholder="Ask about smartphones, privacy, or terms...")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if st.button("Submit Query"):
        if (not csv_upload and not csv_url.strip()):
            st.error("Please provide a CSV file via upload or URL.")
            return
        if not privacy_upload or not terms_upload:
            st.error("Please upload both the Privacy Policy and Terms & Conditions files.")
            return
        if not api_key_input.strip():
            st.error("Please provide a valid API key.")
            return
        if not user_query.strip():
            st.error("Please enter a question.")
            return
        
        with st.spinner("Processing documents..."):
            docs = prepare_documents(csv_upload, csv_url, privacy_upload, terms_upload)
            if not docs:
                st.error("No documents could be processed.")
                return
            split_docs = split_documents(docs)
            vector_store = create_vector_store(split_docs)
            qa_chain = build_qa_chain(api_key_input, vector_store)
        
        history = "\n".join(st.session_state.conversation_history[-5:])
        with st.spinner("Generating answer..."):
            result = qa_chain({"question": user_query})  # Only pass question here
            answer = result["answer"]
        st.markdown("**Answer:**")
        st.write(answer)
        st.session_state.conversation_history.append(f"User: {user_query}\nBot: {answer}")

if __name__ == "__main__":
    main()
