from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

def initialize_chatbot():
    model_options = {
        "MediBot (Default)": {"tokenizer": "Omkar7/finetuned-t5-medical", "model": "Omkar7/finetuned-t5-medical"},
        "MediBot (Llama)": {"tokenizer": "Omkar7/Llama-2-7b-chat-finetune-DPM2009-v2", "model": "Omkar7/Llama-2-7b-chat-finetune-DPM2009-v2"},
        "MediBot (Mistral)": {"tokenizer": "Omkar7/mistral-7b-finetuned-DPM2009", "model": "Omkar7/mistral-7b-finetuned-DPM2009"},
    }

    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

    # Load selected model
    tokenizer = AutoTokenizer.from_pretrained(model_options[selected_model]["tokenizer"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_options[selected_model]["model"])

    return tokenizer, model

def process_user_input(prompt, tokenizer, model):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=250)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response