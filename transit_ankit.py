import streamlit as st
from pdf2image import convert_from_bytes
import cv2, numpy as np
from transformers import (
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
import torch
from datasets import load_dataset
from PIL import Image
import pytesseract
import easyocr

# 1. Cached OCR resources
@st.cache_resource
def load_ocr_components():
    feature_extractor = ViTImageProcessor.from_pretrained(
        "microsoft/trocr-small-printed"
    )  # Image→tensor converter :contentReference[oaicite:4]{index=4}
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/trocr-small-printed", use_fast=True
    )
    try:
        model = VisionEncoderDecoderModel.from_pretrained("./trocr-transit")
    except:
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-printed"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return feature_extractor, tokenizer, model, device

# 2. Cached dataset loader
@st.cache_resource
def load_transit_dataset(csv_file: str):
    ds = load_dataset("csv", data_files={
        "train": csv_file,
        "validation": csv_file.replace("train", "val")
    })
    return ds["train"], ds["validation"]

# 3. Image preprocessing
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None: return image
    angles = [(rho_theta[0][1] - np.pi/2) * (180/np.pi) for rho_theta in lines]
    angle = np.median(angles)
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255,255,255))  # :contentReference[oaicite:5]{index=5}

def preprocess_image(pil_img):
    # Deskew & grayscale
    img = np.array(pil_img)
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)  # :contentReference[oaicite:6]{index=6}
    # Adaptive threshold + denoise
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )  # :contentReference[oaicite:7]{index=7}
    den = cv2.medianBlur(bw, 3)
    # Stack to RGB
    rgb = np.stack([den]*3, axis=-1)  # shape (H,W,3) :contentReference[oaicite:8]{index=8}
    return Image.fromarray(rgb)

# 4. Fine-tuning preprocessing
def preprocess_batch(batch):
    imgs = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    clean = [preprocess_image(img) for img in imgs]
    enc = processor(images=clean, return_tensors="pt",
                    padding="max_length", truncation=True)
    labels = processor.tokenizer(
        batch["text"], padding="max_length",
        truncation=True, return_tensors="pt"
    ).input_ids
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    return {"pixel_values": enc.pixel_values, "labels": labels}

# 5. OCR with display and ensemble
def perform_ocr(pdf_bytes, dpi=300):
    fe, tok, model, device = load_ocr_components()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)  # :contentReference[oaicite:9]{index=9}
    outputs = []
    for i, pg in enumerate(pages):
        st.subheader(f"Preprocessed Page {i+1}")
        clean = preprocess_image(pg)
        st.image(clean, caption=f"Page {i+1}", use_column_width=True)  # :contentReference[oaicite:10]{index=10}
        pix = fe(images=clean, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pix)
        txt = tok.batch_decode(ids, skip_special_tokens=True)[0]
        if len(txt.strip()) < 10:
            txt = pytesseract.image_to_string(np.array(clean))  # :contentReference[oaicite:11]{index=11}
            if not txt.strip():
                reader = easyocr.Reader(['en'])
                txt = "\n".join([res[1] for res in reader.readtext(np.array(clean))])  # :contentReference[oaicite:12]{index=12}
        outputs.append(f"Page {i+1}:\n{txt}")
    return "\n\n".join(outputs)

# 6. Streamlit UI
st.title("Advanced Transit PDF OCR & Fine-Tuning")

# OCR section
uploaded = st.file_uploader("Upload PDF for OCR", type="pdf")
if uploaded:
    with st.spinner("Extracting..."):
        result = perform_ocr(uploaded.read())
    st.subheader("OCR Results")
    st.text_area("Extracted Text", result, height=300)
    st.download_button("Download Text", result, "ocr.txt", "text/plain")

# Fine-tuning section
processor, _, model, _ = load_ocr_components()
csv_path = st.text_input("Train manifest CSV path", "train.csv")
if st.button("Train Model"):
    st.info("Fine-tuning in progress…")
    train_ds, val_ds = load_transit_dataset(csv_path)
    train_ds = train_ds.map(preprocess_batch, batched=True)
    val_ds   = val_ds.map(preprocess_batch, batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir="./trocr-transit", per_device_train_batch_size=2,
        per_device_eval_batch_size=2, predict_with_generate=True,
        evaluation_strategy="steps", eval_steps=200,
        logging_steps=50, num_train_epochs=5,
        learning_rate=3e-5, save_steps=500, save_total_limit=2
    )
    collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.feature_extractor
    )
    trainer.train()
    st.success("Fine-tuning complete and model saved!")
