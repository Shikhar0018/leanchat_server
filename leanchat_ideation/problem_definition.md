# LeanChat Project Deliverables

This document contains all the deliverables for our Business Chatbot Proof-of-Concept (POC). The project comprises both a backend server (implemented in Flask) and a frontend UI (built with Streamlit) that answers queries based on provided product data (CSV), privacy policy, and terms & conditions documents. Additionally, our submission includes a CV and the results from the Principles You personality test.

---

## 1. Problem Definition Document

**Overview:**  
The business is experiencing significant customer support challenges:
- **Long Response Times:** Customers face extended waiting periods.
- **Reduced Satisfaction:** Low customer satisfaction metrics due to delayed responses.
- **High Cart Abandonment:** Extended delays contributing to increased cart abandonment.
- **Multilingual Support Gap:** Existing systems do not adequately support multilingual queries (especially Arabic).

**Data Sources & Assumptions:**  
- **Product Data:** Provided as a CSV file with semicolon-delimited fields.
- **Legal Documentation:** Privacy Policy and Terms & Conditions are supplied in PDF or TXT format and may include content in Arabic.
- **User Behavior:** Queries are expected to relate only to the product data and legal documents; out-of-scope queries should be politely declined.
- **Business Impact:**  
  - Faster responses can increase customer satisfaction.
  - Reduced human support workload.
  - Improved compliance through strict adherence to provided legal documentation.

---

## 2. Solution Design Document

**Proposed Architecture & Technology Stack:**

- **Frontend/UI:**  
  - **Streamlit:** For interactive web-based UI demonstration.
  
- **Backend:**  
  - **Flask:** REST API server handling file uploads, CSV URL settings, API key storage, and chat queries.
  - **LangChain:** For document ingestion, prompt chaining, and integrating with the language model.
  - **FAISS:** For indexing and querying document embeddings.
  - **HuggingFace Embeddings:** Utilizes the "sentence-transformers/all-MiniLM-L6-v2" model for free text embeddings.
  - **Google Gemini API:** Wrapped within a custom LangChain LLM to generate responses.

- **Document Processing:**  
  - File uploads and CSV URL support.
  - **PyPDF2:** For extracting text from PDF files.
  - **Pandas:** For processing CSV files.

- **Conversation Management:**  
  - Maintains the last 5 interactions as conversation history.
  - Custom prompt template enforces responses only within the provided context.

**Flow Diagram:**

          +---------------------+
          |    User Query       |
          +----------+----------+
                     |
                     v
     +---------------+----------------+
     |  Streamlit Frontend UI         |
     +---------------+----------------+
                     |
                     v
     +---------------+----------------+
     |   Flask Backend API            |
     | - File Uploads & CSV URL       |
     | - API Key Storage              |
     | - Chat Query Processing        |
     +---------------+----------------+
                     |
                     v
     +---------------+----------------+
     | Document Processing Pipeline   |
     | (CSV, PDF/TXT Ingestion, Splitting)|
     +---------------+----------------+
                     |
                     v
     +---------------+----------------+
     |   FAISS Vector Store           |
     |   (with HuggingFace Embeddings)|
     +---------------+----------------+
                     |
                     v
     +---------------+----------------+
     |   LangChain & RetrievalQA      |
     |   with Custom Prompt           |
     +---------------+----------------+
                     |
                     v
          +---------------------+
          | Google Gemini API   |
          +---------------------+
                     |
                     v
          +---------------------+
          |  Generated Answer   |
          +---------------------+

---

## 3. Functional Chatbot Prototype

**Description:**  
The prototype is a working POC that demonstrates core functionality:
- **Backend Server:** Implemented in Flask (`backend.py`), exposing endpoints for file uploads, CSV URL setting, API key storage, and chat queries.
- **Frontend UI:** Built using Streamlit (`app.py`) for interactive testing.
- **Core Features:**
  - Upload product CSV data or provide a URL.
  - Upload privacy policy and terms documents (supports PDF/TXT in Arabic and English).
  - Process documents into searchable context using LangChain and FAISS.
  - Custom prompt enforces responses only based on provided context.
  - Maintains a conversation history for context-aware responses.
  - Generates answers via Google Gemini API integration.

**Usage:**  
- Run the backend server: `python backend.py`
- Run the Streamlit UI: `streamlit run app.py`

---

## 4. Demo Presentation

**Outline:**  
- **Introduction (1 minute):**  
  - Overview of the problem and business impact.
  - Brief on the solution's purpose.

- **Architecture & Technology Stack (2 minutes):**  
  - Present the flow diagram.
  - Explain the chosen technology stack (Flask, Streamlit, LangChain, FAISS, HuggingFace, Google Gemini API).

- **Live Demo (2-3 minutes):**  
  - Demonstrate file uploads (CSV, privacy, terms).
  - Show a sample query and the chatbot’s response.
  - Highlight the conversation history feature.

- **Evaluation & Next Steps (1-2 minutes):**  
  - Present evaluation metrics.
  - Discuss additional features and future enhancements.

**Format:**  
Prepare a 5-10 minute slide deck (PowerPoint, Google Slides, or similar) with screenshots and a live demo recording.

---

## 5. Evaluation Plan

**Metrics to Evaluate the POC:**
- **Response Time:** Target response < 5 seconds per query.
- **Accuracy & Relevance:** User surveys and A/B testing to measure how well responses adhere to the provided context.
- **User Engagement:** Track conversation length, number of queries per session, and user feedback.
- **Compliance:** Percentage of out-of-scope queries correctly declined.
- **System Stability:** Monitor API uptime, error logs, and document processing success.

---

## 6. Additional Features List

**Potential Additional Features:**
1. **Multi-Channel Integration:**  
   - Integration with popular messaging platforms (e.g., Slack, WhatsApp).
   - *Rationale:* Increase accessibility and user engagement.
2. **Enhanced Language Support:**  
   - Extend support to additional languages and regional dialects.
   - *Rationale:* Broaden market reach.
3. **User Authentication & Session Management:**  
   - Implement user accounts and persistent conversation history.
   - *Rationale:* Provide personalized experiences and long-term engagement.
4. **Feedback Collection:**  
   - Integrate a mechanism for users to rate responses.
   - *Rationale:* Continuous improvement through user feedback.
5. **Advanced Analytics Dashboard:**  
   - Real-time monitoring of chatbot performance and user interactions.
   - *Rationale:* Optimize performance and track KPIs.

---

## 7. CV

**Attached:**  
Please find my Curriculum Vitae (CV) attached as a separate document (CV.pdf). It outlines my experience, skills, and relevant achievements in the field.

---

## 8. Principles You Results

**Attached:**  
Also included are the results from the Principles You personality test (PrinciplesYou_Results.pdf). These results provide insights into my personal strengths and problem-solving approach, which have informed the design and development of this chatbot solution.

---

## How to Access the Code

- **Backend Server Repository:**  
  [https://github.com/Shikhar0018/leanchat_server](https://github.com/Shikhar0018/leanchat_server)
- **Frontend Code Repository:**  
  [https://github.com/Shikhar0018/mena-chat-partner](https://github.com/Shikhar0018/mena-chat-partner)

You can download the complete code by clicking the "Download ZIP" option on GitHub.

---

## Contact Information

For any questions or further information regarding these deliverables, please contact:

**Name:** Shikhar Jain  
**Email:** shikharjain2001@gmail.com  
<!-- **Phone:** +91 9717393656 -->

---

Thank you for reviewing our project.
