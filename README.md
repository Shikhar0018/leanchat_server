```markdown
# Business Chatbot Backend Server

This repository contains the backend server for a business chatbot proof-of-concept (POC). The chatbot is designed to answer queries based on provided data – a CSV file containing product information, and documents for the privacy policy and terms & conditions. The backend is implemented using Flask and integrates with LangChain to process the documents, build a vector store using HuggingFace embeddings, and call the Google Gemini API to generate responses. A simple Streamlit-based UI (app.py) is also provided to demonstrate the chatbot functionality.

## Repository Structure

- **backend.py**: Contains the complete Flask server code with API endpoints for:
  - File uploads and CSV URL setting
  - API key storage
  - Chat interactions (including conversation history)
  - Health check and status endpoints
- **app.py**: Contains a Streamlit UI for interacting with the backend POC.
- **README.md**: This file.

## Features

- **File Uploads & CSV URL:**  
  Users can upload a CSV file containing product data or provide a CSV URL. Additionally, they can upload a privacy policy and terms & conditions document (PDF or TXT, including Arabic content).

- **Document Processing:**  
  Uploaded files are processed into documents, split into chunks, and indexed using a FAISS vector store with HuggingFace embeddings.

- **Chatbot Integration:**  
  The chatbot uses a custom prompt template to restrict answers to the provided context. It maintains a conversation history (last 5 interactions) to improve context.

- **Google Gemini API Integration:**  
  The backend calls the Google Gemini API (wrapped in a custom LangChain LLM) to generate responses based on user queries.

## API Endpoints

### Health Check
- **GET** `/health`  
  Returns a simple JSON object indicating the server is running.
  ```json
  { "status": "ok" }
  ```

### File Upload
- **POST** `/upload`  
  Upload a file (CSV, privacy, or terms) via form data.  
  **Parameters:**  
  - `type`: Must be one of `csv`, `privacy`, or `terms`.
  - `file`: The file to upload.  
  **Response:**  
  Returns a JSON object indicating success and a summary of document processing.
  ```json
  { "success": true, "results": { "num_docs": 3, "num_chunks": 25 } }
  ```

### Set CSV URL
- **POST** `/set-csv-url`  
  Set the URL for a CSV file.  
  **Payload:**
  ```json
  { "url": "https://example.com/data.csv" }
  ```  
  **Response:**
  ```json
  { "success": true }
  ```

### Files Status
- **GET** `/files-status`  
  Returns the upload status of CSV, privacy, and terms files.
  ```json
  {
      "csv": true,
      "privacy": true,
      "terms": true
  }
  ```

### API Key Storage
- **POST** `/api-key`  
  Stores the Google Gemini API key for use in chat queries.  
  **Payload:**
  ```json
  { "api_key": "YOUR_GEMINI_API_KEY" }
  ```  
  **Response:**
  ```json
  { "success": true }
  ```

### Chat Query
- **POST** `/chat`  
  Sends a chat message to the chatbot. The server retrieves the relevant documents and conversation history, calls the Gemini API, and returns the generated response.  
  **Payload:**
  ```json
  { "message": "What is the warranty period for the smartphone?" }
  ```  
  **Response:**
  ```json
  { "response": "Based on the provided context, the warranty period is ..." }
  ```

### Clear API Key
- **POST** `/clear-api-key`  
  Clears the stored Google Gemini API key.  
  **Response:**
  ```json
  { "success": true }
  ```

### Conversation History
- **GET** `/history`  
  Returns the conversation history (stored in memory) for debugging and tracking.
  ```json
  { "history": ["User: ...", "Bot: ..."] }
  ```

## Running the Project

1. **Clone the Repository:**

   ```sh
   git clone <YOUR_GIT_URL>
   cd <YOUR_PROJECT_NAME>
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8+ installed, then run:

   ```sh
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt`, install these packages manually:

   ```sh
   pip install Flask flask-cors pandas requests PyPDF2 langchain faiss-cpu sentence-transformers streamlit
   ```

3. **Run the Backend Server:**

   Start the backend server (runs on port 9000):

   ```sh
   python backend.py
   ```

4. **Run the Streamlit UI (Optional):**

   In a separate terminal, start the Streamlit UI:

   ```sh
   streamlit run app.py
   ```

## Usage

- **Uploading Files:**  
  Use the `/upload` endpoint to upload your CSV file, privacy policy, and terms documents.
  
- **Setting CSV URL:**  
  Use the `/set-csv-url` endpoint to provide a CSV URL if preferred.
  
- **Storing API Key:**  
  Use the `/api-key` endpoint to store your Google Gemini API key.
  
- **Chat Interaction:**  
  Use the `/chat` endpoint to send your queries. The server uses document context and conversation history to generate accurate responses.
  
- **Debugging:**  
  Check the `/files-status` and `/history` endpoints to view the current status of uploaded files and conversation history.

## Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
