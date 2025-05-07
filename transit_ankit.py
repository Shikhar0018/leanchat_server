import streamlit as st
import pdfplumber                                              # PDF parsing :contentReference[oaicite:5]{index=5}
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, AutoProcessor, AutoModelForDocumentQuestionAnswering
import json, re, numpy as np
from PIL import Image                                          # For page images

# ── Configuration ───────────────────────────────────────────────────────────────
NER_MODEL_NAME      = "dslim/distilbert-NER"                    # ~261 MB, F1≈0.92 on CoNLL‑2003 :contentReference[oaicite:6]{index=6}
PDF_QA_MODEL        = "microsoft/layoutlmv3-base"               # LayoutLMv3 for Doc‑QA :contentReference[oaicite:7]{index=7}
GEN_MODEL           = "Qwen/Qwen1.5-0.5B-Chat"                   # No SentencePiece, CPU‑runnable :contentReference[oaicite:8]{index=8}

# ── PDF → text+words+boxes+image ─────────────────────────────────────────────────
def extract_pages(uploaded_file):
    """Return list of dicts: {text, words, boxes, image} for each page."""
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            # 1) raw text (layout‑aware) :contentReference[oaicite:9]{index=9}
            txt = page.extract_text(layout=True)
            # 2) word‑level OCR boxes :contentReference[oaicite:10]{index=10}
            wdicts = page.extract_words()
            words = [w["text"] for w in wdicts]
            boxes = [[w["x0"], w["top"], w["x1"], w["bottom"]] for w in wdicts]
            # 3) render page as PIL image :contentReference[oaicite:11]{index=11}
            pil_img = page.to_image(resolution=300).original
            pages.append({"text": txt, "words": words, "boxes": boxes, "image": pil_img})
    return pages

# ── Load pipelines once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_ner():
    tok = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    mdl = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple", device=-1)

@st.cache_resource
def load_pdf_qa():
    # 1. Build processor (image_processor + tokenizer)
    processor = AutoProcessor.from_pretrained(PDF_QA_MODEL, apply_ocr=True)
    # 2. Load the extractive QA model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(PDF_QA_MODEL)
    # 3. Create the pipeline with both tokenizer & feature_extractor
    return pipeline(
        "document-question-answering",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.image_processor,
        device=-1
    )

@st.cache_resource
def load_gen():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(GEN_MODEL, trust_remote_code=True)
    return pipeline("text-generation", model=mdl, tokenizer=tok, device=-1, max_new_tokens=256)

# ── JSON parsing helpers ─────────────────────────────────────────────────────────
def safe_float(v):
    try: return float(str(v).replace(",", "").split()[0])
    except: return 0.0

def safe_int(v):
    try: return int("".join(filter(str.isdigit, str(v))))
    except: return 0

def create_default():
    return {
      "company_name":None,"customer_name":None,"delivery_address":None,
      "date":None,"cin_number":None,"pan_number":None,"gst_number":None,
      "total_challan_value":0.0,"weights":{"net":0.0,"gross":0.0,"tare":0.0},
      "trip_number":0,"delivery_customer_number":None
    }

# ── Top‑level parsing of PDF via Doc‑QA ──────────────────────────────────────────
def process_pdf(pages):
    """
    Ask key questions of the first page image.
    The pipeline will:
      1) run OCR on the image,
      2) tokenize words + boxes,
      3) forward pixel_values + bbox + input_ids into the model.
    """
    qa = load_pdf_qa()
    img = pages[0]["image"]   # a PIL.Image from pdfplumber

    out = {}
    questions = {
        "company_name":        "What is the company name?",
        "total_challan_value": "What is the total challan value?",
        "weights":             "What are the net, gross, and tare weights?",
        "gst_number":          "What is the GST number?",
        "trip_number":         "What is the trip number?"
    }

    for key, ques in questions.items():
        # ← correct kwarg name is `image=`
        answers = qa(image=img, question=ques)
        out[key] = answers[0]["answer"] if answers else None

    # post‑process weights into sub‑fields
    if out.get("weights"):
        nums = re.findall(r"[\d,.]+", out["weights"])
        out["weights"] = {
            "net":   float(nums[0].replace(",", "")) if len(nums)>0 else 0.0,
            "gross": float(nums[1].replace(",", "")) if len(nums)>1 else 0.0,
            "tare":  float(nums[2].replace(",", "")) if len(nums)>2 else 0.0,
        }

    return out


# ── Fallback: NER + LLM → JSON ───────────────────────────────────────────────────
def fallback_json(ocr_lines, entities):
    gen = load_gen()
    prompt = (
      "### Instruction:\n"
      "Convert these OCR lines + entities into STRICT JSON with keys:\n"
      "company_name, customer_name, delivery_address, date, cin_number, pan_number,\n"
      "gst_number, total_challan_value, weights (net, gross, tare), trip_number,\n"
      "delivery_customer_number.\n\n"
      f"OCR lines: {ocr_lines}\nEntities: {entities}\n\n"
      "### Response (strict JSON, no markdown):"
    )
    txt = gen(prompt, do_sample=False)[0]["generated_text"]
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return create_default()
    try:
        obj = json.loads(m.group(0))
        return obj
    except:
        return create_default()

# ── Streamlit UI ────────────────────────────────────────────────────────────────
def main():
    st.title("PDF → Structured JSON (Offline)")
    upload = st.file_uploader("Upload PDF", type="pdf")
    if not upload:
        st.info("Please upload a PDF document.")
        return

    # 1) extract pages
    with st.spinner("Extracting PDF pages…"):
        pages = extract_pages(upload)

    # 2) try document‑QA
    with st.spinner("Running Document QA…"):
        qa_res = process_pdf(pages)

    # If any key is missing, fall back on NER+LLM
    if any(qa_res[k] in (None, "") for k in qa_res):
        ocr_txt = [p["text"] for p in pages]
        full = " ".join(ocr_txt)
        ents = load_ner()(full)
        qa_res = fallback_json(ocr_txt, ents)

    st.subheader("Structured Output")
    st.json(qa_res)

if __name__ == "__main__":
    main()
