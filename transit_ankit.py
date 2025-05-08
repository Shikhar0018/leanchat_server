import streamlit as st
from pdf2image import convert_from_bytes
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

@st.cache_resource
def load_ocr_components():
    # Load components separately with explicit settings
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-small-printed")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return feature_extractor, tokenizer, model, device

def perform_ocr(pdf_bytes, dpi=200):
    """Perform OCR using VisionEncoderDecoder with GPT2 tokenizer"""
    feature_extractor, tokenizer, model, device = load_ocr_components()
    
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    full_text = []
    
    for idx, image in enumerate(images):
        st.progress((idx + 1) / len(images))
        
        # Process image through feature extractor
        pixel_values = feature_extractor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(device)
        
        # Generate text with tokenizer
        generated_ids = model.generate(pixel_values)
        page_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        full_text.append(f"Page {idx+1}:\n{page_text}\n\n")
    
    return "\n".join(full_text)

# Streamlit UI
st.title("PDF OCR with GPT2 Tokenizer")
st.markdown("Upload a PDF file for text extraction")

uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_content = uploaded_file.read()
        extracted_text = perform_ocr(pdf_content)
        
        st.subheader("Extracted Text")
        st.text_area("OCR Output", extracted_text, height=400)
        
        st.download_button(
            "Download Text",
            extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )