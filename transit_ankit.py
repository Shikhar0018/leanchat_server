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
    # Clean the output and convert to valid JSON

    # Sample Output of raw_text: '{\'entity_group\': \'ORG\', \'score\': np.float32(0.99845433), \'word\': \'ZAPRAXAIR\', \'start\': 1, \'end\': 10}, {\'entity_group\': \'ORG\', \'score\': np.float32(0.9995516), \'word\': \'Praxair\', \'start\': 34, \'end\': 41}, {\'entity_group\': \'ORG\', \'score\': np.float32(0.9963855), \'word\': \'India Private Limited\', \'start\': 45, \'end\': 66}]\n\n"\n        "### Response (strict JSON, no markdown):"\n\nOutput STRICT JSON (no text) with ALL fields. Use null for missing values.\n{null,null,"Praxair","India Private Limited",null,"CIN No U24111KA1996PTC020272","PAN No AAACPS9993J","GST No 33AAACP9993J1ZV","Customer Delivery Address","PROTO METALICS PRIVATE LIMITED","Plot No.K-20 & 21, South Venue","Date 03.01.2025","Total Challan Value 17,652.80","Net Weight 5.000 KG","Gross Weight 14,100 KG","Tare Weight 9100 KG","Trip No. 0001117935","Delivery Customer No 3197104"] The OCR lines have been successfully converted to a structured JSON object with the specified keys and detected entities. Here is the output:\n\n```json\n{\n    "company_name": "Praxair",\n    "customer_name": "India Private Limited",\n    "delivery_address": "Proto Metalics Private Limited",\n    "date": "03.01.2025",\n    "cin_number": "U24111KA1996PTC020272",\n    "pan_number": "AAACPS9993J",\n    "gst_number": "33AAACP9993J1ZV",\n    "total_challan_value": "17,652.80",\n    "weights": [\n        {"net_weight": "5.000 KG"},\n        {"gross_weight": "14,100 KG"},\n        {"tare_weight": "9100 KG"}\n    ],\n    "trip_number": "0001117935",\n    "delivery_customer_number": "3197104"\n}\n```\n\nThis JSON structure accurately represents all the information extracted from the OCR lines while ignoring any non-essential details such as dates or specific identifiers. If you need further customization or additional features, please let me know! Let\'s proceed with converting the next line of OCR lines into a structured JSON object.\n
    # Make use of nlp model to generate JSON
    # Remove np.float32 and convert to valid JSON

    


    cleaned = re.sub(r"np\.float32\(([\d.]+)\)", r"\1", raw_text)
    cleaned = cleaned.replace("'", '"').replace("null", "null")
    
    # Find JSON block using more robust pattern
    json_match = re.search(r'\{\s*".*?}\s*\}', cleaned, re.DOTALL)
    
    if not json_match:
        return create_default_structure()
    
    try:
        parsed = json.loads(json_match.group(0))
        
        # Convert numeric fields and weights structure
        return {
            "company_name": parsed.get("company_name"),
            "customer_name": parsed.get("customer_name"),
            "delivery_address": parsed.get("delivery_address"),
            "date": parsed.get("date"),
            "cin_number": parsed.get("cin_number"),
            "pan_number": parsed.get("pan_number"),
            "gst_number": parsed.get("gst_number"),
            "total_challan_value": safe_float(parsed.get("total_challan_value", "0").replace(",", "")),
            "weights": parse_weights(parsed.get("weights", [])),
            "trip_number": safe_int(parsed.get("trip_number", "0")),
            "delivery_customer_number": parsed.get("delivery_customer_number")
        }
    except json.JSONDecodeError:
        return create_default_structure()

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