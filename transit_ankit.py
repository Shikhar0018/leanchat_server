import streamlit as st
import pdfplumber
import pytesseract
from transformers import (
    pipeline, AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForCausalLM, AutoProcessor
)
import json
import re
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────
PDF_QA_MODEL = "impira/layoutlm-document-qa"
NER_MODEL = "Babelscape/wikineural-multilingual-ner"
GEN_MODEL = "Qwen/Qwen1.5-0.5B-Chat"

# ── Cached Resources ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "qa": pipeline("document-question-answering", model=PDF_QA_MODEL, device=-1),
        "ner": pipeline("ner", model=NER_MODEL, aggregation_strategy="best", device=-1),
        "gen": pipeline("text-generation", model=GEN_MODEL, device=-1, max_new_tokens=512)
    }

# ── PDF Processing ────────────────────────────────────────────────
def extract_pdf_data(uploaded_file):
    full_text = []
    raw_ocr_pages = []
    
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text with layout preservation
            text = page.extract_text(layout=True) or ""
            if not text.strip():
                # Fallback to OCR for scanned pages
                img = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(img, lang="eng")
            
            full_text.append(text)
            raw_ocr_pages.append({
                "page": page_num + 1,
                "text": text,
                "dimensions": (page.width, page.height)
            })
    
    return " ".join(full_text), raw_ocr_pages

# ── Streamlit UI ──────────────────────────────────────────────────
def main():
    st.title("📄 Industrial Document Processor")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_file:
        # Extract text and show raw OCR
        with st.spinner("Extracting document content..."):
            processed_text, raw_ocr = extract_pdf_data(uploaded_file)
            
            st.subheader("Raw OCR Preview")
            with st.expander("View Raw Extracted Text", expanded=False):
                for page in raw_ocr:
                    st.markdown(f"**Page {page['page']}** ({page['dimensions'][0]}x{page['dimensions'][1]}px)")
                    st.code(page['text'], language="text")
                    st.divider()

        # Process document
        with st.spinner("Analyzing document structure..."):
            models = load_models()
            entities = models["ner"](processed_text)
            qa_results = answer_questions(uploaded_file, models["qa"])
            structured_data = validate_structure({
                **parse_entities(entities),
                **qa_results
            })
        
        # Show results
        st.subheader("Structured Output")
        st.json(structured_data)

def answer_questions(file, qa_pipe):
    questions = {
        "company_name": "What is the company name?",
        "total_challan_value": "What is the total challan value?",
        # ... (rest of the questions)
    }
    return {k: clean_answer(qa_pipe(file, question=v)[0]["answer"]) for k, v in questions.items()}

# ... (rest of the helper functions remain same as previous version)

if __name__ == "__main__":
    main()