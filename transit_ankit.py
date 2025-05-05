import streamlit as st
import google.generativeai as genai
from transformers import pipeline
import json

# Configuration
GENAI_API_KEY = ""  # Set in Streamlit secrets
NER_MODEL_NAME = "Davlan/distilbert-base-multilingual-cased-ner-hrl"  # ~250MB #Rejected
NER_MODEL_NAME_2 = "microsoft/phi-3.5-mini-instruct"  # ~500MB
NER_MODEL_NAME_3 = "meta-llama/Llama-3.2-1B-Instruct"  # ~1.5GB
NER_MODEL_NAME_4 = "Qwen/Qwen2.5-0.5B-Instruct"   # ~187MB
# NER_MODEL_NAME = "microsoft/phi-3.5-mini-instruct"  

def process_entities(text):
    ner_pipeline = pipeline(
        "ner",
        model=NER_MODEL_NAME,
        tokenizer=NER_MODEL_NAME,
        aggregation_strategy="simple"
    )
    entities = ner_pipeline(text)
    return entities

generator = pipeline(
    "text-generation",
    model= NER_MODEL_NAME_4,      # local instruct‑tuned model :contentReference[oaicite:4]{index=4}
    tokenizer=  NER_MODEL_NAME_4,
    device="mps",                                    # use GPU (>=0) or CPU (-1)
)

def generate_structured_data_locally(ocr_lines, entities):
    # Build a single prompt instructing JSON output
    prompt = (
        "### Instruction:\n"
        "Convert these OCR lines into a structured JSON object with keys:\n"
        "company_name, customer_name, delivery_address, date, cin_number, pan_number, "
        "gst_number, total_challan_value, weights (net, gross, tare), trip_number, delivery_customer_number.\n\n"
        f"OCR lines: {ocr_lines}\n"
        f"Detected entities: {entities}\n\n"
        "### Response (strict JSON, no markdown):"
    )

    # Generate with a stop sequence to enforce JSON only :contentReference[oaicite:5]{index=5}
    output = generator(
        prompt,
        max_length=512,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
        # You can add `stop_sequences=["\n\n"]` if supported in your transformers version
    )
    text = output[0]["generated_text"]
    # Extract the JSON substring (assumes the model obeys the prompt)
    json_start = text.find("{")
    json_text = text[json_start:]
    return json_text

def generate_structured_data(ocr_lines, entities):
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""Convert these OCR lines into structured JSON. 
    OCR lines: {ocr_lines}
    Detected entities: {entities}
    
    Extract following fields:
    - company_name
    - customer_name
    - delivery_address
    - date
    - cin_number
    - pan_number
    - gst_number
    - total_challan_value
    - weights (net, gross, tare)
    - trip_number
    - delivery_customer_number
    
    Output ONLY valid JSON without markdown formatting."""
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("OCR Data Structuring")
    
    # Input for OCR lines
    ocr_input = st.text_area("Paste OCR lines (one per line)", height=200)
    
    if ocr_input:
        ocr_lines = [line.strip() for line in ocr_input.split('\n') if line.strip()]
        full_text = " ".join(ocr_lines)
        
        # Entity extraction
        with st.spinner("Analyzing content..."):
            entities = process_entities(full_text)
        
        st.subheader("Detected Entities")
        st.write(entities)
        
        if st.button("Generate Structured Data"):
            with st.spinner("Generating JSON structure..."):
                try:
                    # json_output = generate_structured_data(ocr_lines, entities)
                    json_output = generate_structured_data_locally(ocr_lines, entities)
                    parsed_json = json.loads(json_output)
                    
                    st.subheader("Structured Output")
                    st.json(parsed_json)
                    
                except Exception as e:
                    st.error(f"Error generating JSON: {str(e)}")
                    st.write("Raw response:", json_output)

if __name__ == "__main__":
    main()