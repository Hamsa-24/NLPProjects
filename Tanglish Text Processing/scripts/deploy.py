import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_dir = "models/trained_model/"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

st.title("Tanglish to Tamil Transliterator")
text_input = st.text_input("Enter Tanglish text:")

if text_input:
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    transliterated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Transliterated Tamil Text: {transliterated_text}")
