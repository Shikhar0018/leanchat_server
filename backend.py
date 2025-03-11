import os
import io
import csv
import uuid
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# In-memory storage for demonstration (use a database in production)
session_data = {
    'vector_store': None,
    'conversation_history': [],
    'uploaded_files': {
        'csv': None,
        'privacy': None,
        'terms': None
    },
    'csv_url': None,
    'gemini_api_key': None  # API key storage
}

# ------------------------------------------------------------------------------
# File processing functions
def csv_from_url_to_dict(url: str) -> List[dict]:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching the URL: {url}")
    # Use errors="replace" to handle decoding issues
    csv_content = response.content.decode('utf-8', errors="replace")
    csv_file = io.StringIO(csv_content)
    data_dict = []
    # Specify delimiter if needed; adjust if your CSV uses a different delimiter.
    csv_reader = csv.DictReader(csv_file, delimiter=";")
    data_dict = list(csv_reader)
    return data_dict

def docs_from_dicts(dict_list: List[dict]) -> List[Document]:
    docs = []
    for item in dict_list:
        content = "\n".join([f"{k}: {v}" for k, v in item.items()])
        docs.append(Document(page_content=content))
    return docs

def load_csv_file(content: bytes) -> List[Document]:
    try:
        # Decode bytes to string with proper encoding handling
        content_str = content.decode('utf-8-sig', errors="replace")
        
        # Create pandas DataFrame from string content
        df = pd.read_csv(io.StringIO(content_str))
        
        docs = []
        for _, row in df.iterrows():
            # Preserve special characters in text
            text = row.to_string(index=False)
            docs.append(Document(page_content=text))
        return docs
    
    except Exception as e:
        app.logger.error(f"CSV loading error: {str(e)}")
        return []

def load_text_file(content: bytes, filename: str) -> Document:
    try:
        # Try UTF-8-SIG for BOM encoded files
        text = content.decode('utf-8-sig', errors="replace")
    except UnicodeDecodeError:
        text = content.decode('utf-8', errors="replace")

    if filename.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            text = f"Error reading PDF: {e}"
    else:
        text = content.decode('utf-8', errors="replace")
    return Document(page_content=text)

# ------------------------------------------------------------------------------
# Document processing pipeline
def process_documents():
    docs = []
    
    # Process CSV data
    if session_data['csv_url']:
        try:
            dict_list = csv_from_url_to_dict(session_data['csv_url'])
            docs.extend(docs_from_dicts(dict_list))
        except Exception as e:
            app.logger.error(f"Error loading CSV from URL: {e}")
    elif session_data['uploaded_files']['csv']:
        docs.extend(load_csv_file(session_data['uploaded_files']['csv']))
    
    # Process privacy policy file
    if session_data['uploaded_files']['privacy']:
        val = session_data['uploaded_files']['privacy']
        if isinstance(val, tuple):
            content, filename = val
        else:
            content, filename = val, "privacy.txt"
        docs.append(load_text_file(content, filename))
    
    # Process terms & conditions file
    if session_data['uploaded_files']['terms']:
        val = session_data['uploaded_files']['terms']
        if isinstance(val, tuple):
            content, filename = val
        else:
            content, filename = val, "terms.txt"
        docs.append(load_text_file(content, filename))
    
    # Split documents
    # Modify the text splitter in process_documents()
    splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,  # Reduced for better handling of Arabic script
    chunk_overlap=50,
    strip_whitespace=False  # Important for preserving Arabic text formatting
    )

    split_docs = []
    for doc in docs:
        split_docs.extend(splitter.split_text(doc.page_content))
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    session_data['vector_store'] = FAISS.from_documents(
        [Document(page_content=t) for t in split_docs], 
        embeddings
    )
    return { "num_docs": len(docs), "num_chunks": len(split_docs) }

# ------------------------------------------------------------------------------
# Custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a multilingual consumer chatbot for our smartphone business. "
        "You must answer in the same language as the question. "
        "If the question is in Arabic, respond in Arabic. "
        "If in English, respond in English.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

# ------------------------------------------------------------------------------
# Google Gemini LLM wrapper
class GoogleGemini(LLM):
    api_key: str = ""
    
    def __init__(self, api_key: str, **kwargs):
        # Call the parent initializer
        super().__init__(**kwargs)
        # Set the api_key field using object.__setattr__
        object.__setattr__(self, "api_key", api_key)
        # Manually mark the field as set in the pydantic model
        object.__setattr__(self, "__pydantic_fields_set__", {"api_key"})
    
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
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                json_response = response.json()
                try:
                    result = json_response["candidates"][0]["content"]["parts"][0]["text"]
                    return result
                except Exception as e:
                    app.logger.error(f"Gemini API Error: {str(e)}")
                    app.logger.debug(f"Request payload: {data}")
                    return "Error parsing API response"
            else:
                return f"API error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Gemini API Error: {str(e)}")
            app.logger.debug(f"Request payload: {data}")
            return f"API request failed: {str(e)}"

    def predict(self, prompt: str, **kwargs) -> str:
        return self._call(prompt, kwargs.get("stop"))


# ------------------------------------------------------------------------------
# Build the RetrievalQA chain using Google Gemini LLM with custom prompt
def build_qa_chain(api_key: str, vector_store) -> RetrievalQA:
    print("Building QA chain")
    gemini_llm = GoogleGemini(api_key=api_key)
    
    return RetrievalQA.from_chain_type(
        llm=gemini_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": custom_prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        }
    )


# ------------------------------------------------------------------------------
# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/upload', methods=['POST'])
def handle_upload():
    file_type = request.form.get('type')
    file = request.files.get('file')
    
    if not file or file_type not in ['csv', 'privacy', 'terms']:
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        content = file.read()  # Read file as bytes
        # Store file content (as bytes) in session_data
        session_data['uploaded_files'][file_type] = content
        results = process_documents()
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set-csv-url', methods=['POST'])
def set_csv_url():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({"error": "Missing URL"}), 400
    
    session_data['csv_url'] = data['url']
    try:
        process_documents()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files-status', methods=['GET'])
def check_files_status():
    return jsonify({
        "csv": bool(session_data['csv_url'] or session_data['uploaded_files']['csv']),
        "privacy": bool(session_data['uploaded_files']['privacy']),
        "terms": bool(session_data['uploaded_files']['terms'])
    }), 200

@app.route('/api-key', methods=['POST'])
def save_api_key():
    data = request.json
    if not data or 'api_key' not in data:
        return jsonify({"error": "Missing API key"}), 400
    
    session_data['gemini_api_key'] = data['api_key']
    return jsonify({"success": True})

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    if not session_data['vector_store']:
        return jsonify({"error": "No documents loaded"}), 400

    if not session_data['gemini_api_key']:
        return jsonify({"error": "API key not configured"}), 401

    try:
        # Build the QA chain
        qa_chain = build_qa_chain(
            session_data['gemini_api_key'],
            session_data['vector_store']
        )

        # Get conversation history and format question
        history = "\n".join(session_data['conversation_history'][-5:])
        formatted_question = (
        f"Conversation History:\n{history}\n\n"
        f"New Question (respond in the same language as this question): {data['message']}"
        )

        # Execute the chain
        result = qa_chain({"query": formatted_question})
        
        # Update conversation history
        session_data['conversation_history'].append(f"User: {data['message']}")
        session_data['conversation_history'].append(f"Bot: {result['result']}")
        
        return jsonify({"response": result['result']})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/clear-api-key', methods=['POST'])
def clear_api_key():
    session_data['gemini_api_key'] = None
    return jsonify({"success": True})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": session_data['conversation_history']}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
