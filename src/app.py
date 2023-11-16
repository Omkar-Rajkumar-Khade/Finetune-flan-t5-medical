import streamlit as st
from generator import initialize_chatbot, process_user_input

# Model initialization
tokenizer, model = initialize_chatbot()

# Streamlit app
st.title("💉 MediBot")
st.write("""MediBot is a comprehensive solution for Medical Question Answering using fine-tuned open-source models""")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to MediBot! I'm Medi, your medical assistant. How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = process_user_input(prompt, tokenizer, model)

    msg = {"role": "assistant", "content": response}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])