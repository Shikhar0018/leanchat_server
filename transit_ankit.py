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
    # Step 1: Clean special characters and model artifacts
    cleaned = re.sub(
        r"(np\.float32\([\d\.]+\)|```json|```|''|\[\]|null)", 
        lambda m: '' if m.group(1) == 'null' else m.group(1),
        raw_text
    )
    cleaned = re.sub(r"'", '"', cleaned)
    
    # Step 2: Extract the most complete JSON block
    json_match = re.search(
        r'(\{[^{}]*(\{[^{}]*\}[^{}]*)*\})', 
        cleaned, 
        re.DOTALL
    )
    
    if not json_match:
        return create_default_structure()
    
    # Step 3: Structural repairs
    json_str = json_match.group(1)
    json_str = re.sub(
        r'"weights":\s*\[([^\]]+)\]', 
        lambda m: '"weights": {' + 
                  m.group(1).replace('"net_weight"', '"net"')
                             .replace('"gross_weight"', '"gross"')
                             .replace('"tare_weight"', '"tare"') + '}', 
        json_str
    )
    
    # Step 4: Convert number-like strings
    json_str = re.sub(
        r'"([\d,]+\.?\d*)"', 
        lambda m: m.group(1).replace(",", ""), 
        json_str
    )
    
    # Step 5: Validate and parse
    try:
        parsed = json.loads(json_str)
    except JSONDecodeError:
        try:
            # Try adding missing quotes
            repaired = re.sub(
                r'([\{,])(\w+)(:)', 
                lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', 
                json_str
            )
            parsed = json.loads(repaired)
        except:
            return create_default_structure()
    
    # Step 6: Enforce schema
    return {
        "company_name": parsed.get("company_name") or extract_entity(parsed, "ORG"),
        "customer_name": parsed.get("customer_name") or extract_entity(parsed, "PER"),
        "delivery_address": format_address(parsed),
        "date": parsed.get("date"),
        "cin_number": extract_pattern(parsed, r"CIN No (\S+)"),
        "pan_number": extract_pattern(parsed, r"PAN No (\S+)"),
        "gst_number": extract_pattern(parsed, r"GST No (\S+)"),
        "total_challan_value": safe_float(parsed.get("total_challan_value")),
        "weights": parse_weights(parsed.get("weights", {})),
        "trip_number": safe_int(parsed.get("trip_number")),
        "delivery_customer_number": parsed.get("delivery_customer_number")
    }

def extract_entity(data, entity_type):
    return next((
        e["word"] for e in data.get("entities", []) 
        if e.get("entity_group") == entity_type
    ), None)

def extract_pattern(text, pattern):
    if isinstance(text, dict):
        text = json.dumps(text)
    match = re.search(pattern, text)
    return match.group(1) if match else None

def format_address(data):
    parts = []
    if "delivery_address" in data:
        parts.append(data["delivery_address"])
    if "address_lines" in data:
        parts.extend(data["address_lines"])
    return ", ".join(parts) if parts else None


def safe_float(value):
    try:
        return float(str(value).replace(",", "").split()[0])
    except:
        return 0.0

def safe_int(value):
    try:
        return int(''.join(filter(str.isdigit, str(value))))
    except:
        return 0

def parse_weights(weights):
    if isinstance(weights, list):
        return {
            "net": safe_float(weights[0].get("net_weight") if weights else 0),
            "gross": safe_float(weights[1].get("gross_weight") if len(weights) > 1 else 0),
            "tare": safe_float(weights[2].get("tare_weight") if len(weights) > 2 else 0)
        }
    return {
        "net": safe_float(weights.get("net", 0)),
        "gross": safe_float(weights.get("gross", 0)),
        "tare": safe_float(weights.get("tare", 0))
    }

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
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"Error generating JSON: {str(e)}")

if __name__ == "__main__":
    main()