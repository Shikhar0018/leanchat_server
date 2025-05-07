import streamlit as st
from transformers import pipeline
import json
import re
from json import JSONDecodeError
import numpy as np


# Configuration
NER_MODEL_NAME = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Updated to better model

def process_entities(text):
    ner_pipeline = pipeline(
        "ner",
        model=NER_MODEL_NAME,
        aggregation_strategy="simple"
    )
    return ner_pipeline(text)

# Load generation pipeline once
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model=GENERATION_MODEL,
        device="mps",
        max_new_tokens=512  # Increased token limit
    )



def parse_generated_json(raw_text):
    # Clean the raw text
    cleaned = re.sub(
        r"(np\.float32\([\d\.]+\)|```json|```|''|\[\]|null|\\n)",
        lambda m: '',
        raw_text
    )
    cleaned = re.sub(r"'", '"', cleaned)
    
    # Advanced JSON extraction with nested pattern matching
    return cleaned
    json_str = None
    deepest_level = 0
    for match in re.finditer(r'(\{([^{}]*|(?R))*\})', cleaned, re.DOTALL):
        candidate = match.group(0)
        brace_diff = candidate.count('{') - candidate.count('}')
        if brace_diff == 0 and candidate.count('{') > deepest_level:
            json_str = candidate
            deepest_level = candidate.count('{')
    
    if not json_str:
        return create_default_structure()
    
    # JSON repair operations
    json_str = (
        json_str
        .replace('", "', '", "')  # Fix missing commas
        .replace(':"', ': "')      # Add space after colons
        .replace(',"', ', "')      # Add space after commas
        .replace('"{', '{')        # Remove quoted braces
        .replace('}"', '}')
    )
    
    # Convert weight array to proper object
    json_str = re.sub(
        r'"weights":\s*\[([^\]]+)\]',
        lambda m: f'"weights": {{{m.group(1).replace("_weight", "")}}}',
        json_str
    )
    
    try:
        parsed = json.loads(json_str)
    except JSONDecodeError:
        # Try to repair malformed JSON
        try:
            parsed = json.loads(re.sub(
                r'("[\w_]+")\s*:',
                lambda m: f'{m.group(1)}:',
                json_str
            ))
        except:
            return create_default_structure()
    
    # Field normalization
    return {
        "company_name": (
            parsed.get("company_name") 
            or extract_value(parsed, ["ORG", "Praxair"])
        ),
        "customer_name": extract_value(parsed, ["customer_name", "ZAPRAXAIR"]),
        "delivery_address": format_address(parsed),
        "date": parse_date(parsed.get("date")),
        "cin_number": extract_pattern(json_str, r"CIN No (\S+)"),
        "pan_number": extract_pattern(json_str, r"PAN No (\S+)"),
        "gst_number": extract_pattern(json_str, r"GST No (\S+)"),
        "total_challan_value": convert_currency(
            parsed.get("total_challan_value")
        ),
        "weights": {
            "net": convert_weight(parsed.get("weights", {}).get("net")),
            "gross": convert_weight(parsed.get("weights", {}).get("gross")),
            "tare": convert_weight(parsed.get("weights", {}).get("tare"))
        },
        "trip_number": safe_int(parsed.get("trip_number")),
        "delivery_customer_number": parsed.get("delivery_customer_number")
    }

# Helper functions
def extract_value(data, keys):
    for key in keys:
        if isinstance(data, dict) and data.get(key):
            return data[key]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get(key):
                    return item[key]
    return None

def extract_pattern(text, regex):
    match = re.search(regex, text)
    return match.group(1) if match else None

def format_address(data):
    address_keys = ["delivery_address", "address", "location"]
    parts = []
    for key in address_keys:
        if data.get(key):
            parts.append(str(data[key]))
    return ", ".join(parts) if parts else None

def convert_currency(value):
    try:
        return float(str(value).replace(",", "").replace("₹", "").strip())
    except:
        return 0.0

def convert_weight(value):
    try:
        return float(str(value).split()[0].replace(",", ""))
    except:
        return 0.0

def parse_date(date_str):
    try:
        return re.sub(r"(\d{2})\.(\d{2})\.(\d{4})", r"\3-\2-\1", str(date_str))
    except:
        return None

def safe_int(value):
    try:
        return int(''.join(filter(str.isdigit, str(value))))
    except:
        return 0

def create_default_structure():
    return {
        "company_name": None,
        "customer_name": None,
        "delivery_address": None,
        "date": None,
        "cin_number": None,
        "pan_number": None,
        "gst_number": None,
        "total_challan_value": 0.0,
        "weights": {"net": 0.0, "gross": 0.0, "tare": 0.0},
        "trip_number": 0,
        "delivery_customer_number": None
    }

def extract_complete_json(text):
    """Improved JSON extraction with validation"""
    try:
        # Find first complete JSON object
        matches = re.findall(r'\{[^{}]*\}', text)
        if matches:
            return json.loads(matches[0])
        return None
    except JSONDecodeError:
        return None

def generate_structured_data_locally(ocr_lines, entities):
    # Improved prompt structure
    prompt = f""" "### Instruction:\n"
        "Convert these OCR lines into a structured JSON object with keys:\n"
        "company_name, customer_name, delivery_address, date, cin_number, pan_number, "
        "gst_number, total_challan_value, weights (net, gross, tare), trip_number, delivery_customer_number.\n\n"
        f"OCR lines: {ocr_lines}\n"
        f"Detected entities: {entities}\n\n"
        "### Response (strict JSON, no markdown):"

Output STRICT JSON (no text) with ALL fields. Use null for missing values.
{{"""
    
    generator = load_generator()
    output = generator(
        prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
        # return_full_text=False  # Prevent prompt repetition
    )
    
    # Extract and validate JSON
    raw_json = "{" + output[0]["generated_text"].split("{", 1)[-1]
    # parsed = extract_complete_json(raw_json)
    parsed = parse_generated_json(raw_json)
    
    # Ensure required fields exist
    # required_fields = {
    #     "company_name": None,
    #     "customer_name": None,
    #     "delivery_address": None,
    #     "date": None,
    #     "cin_number": None,
    #     "pan_number": None,
    #     "gst_number": None,
    #     "total_challan_value": None,
    #     "weights": {
    #         "net": None,
    #         "gross": None,
    #         "tare": None
    #     },
    #     "trip_number": None,
    #     "delivery_customer_number": None
    # }
    
    # if parsed:
    #     for key in required_fields:
    #         if key in parsed:
    #             required_fields[key] = parsed[key]
    #     return required_fields
    return parsed

def main():
    st.title("OCR Data Structuring")
    
    ocr_input = st.text_area("Paste OCR lines (one per line)", height=200)
    
    if ocr_input:
        ocr_lines = [line.strip() for line in ocr_input.split('\n') if line.strip()]
        full_text = " ".join(ocr_lines)
        
        with st.spinner("Analyzing content..."):
            entities = process_entities(full_text)
        
        st.subheader("Detected Entities")
        st.write(entities)
        
        if st.button("Generate Structured Data"):
            with st.spinner("Generating JSON structure..."):
                try:
                    result = generate_structured_data_locally(ocr_lines, entities)
                    st.subheader("Structured Output")
                    # st.json(result)
                    st.write(result)

                    
                except Exception as e:
                    st.error(f"Error generating JSON: {str(e)}")

if __name__ == "__main__":
    main()