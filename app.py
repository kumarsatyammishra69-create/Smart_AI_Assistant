import streamlit as ui
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# ----------- APP CONFIG -----------
ui.set_page_config(page_title="Smart AI Assistant", page_icon="🤖")
ui.header("💬 Smart AI Assistant")

# ----------- STATE MANAGEMENT -----------
if "chat_log" not in ui.session_state:
    ui.session_state.chat_log = []

# ----------- RESET BUTTON -----------
col1, col2 = ui.columns([6, 1])
with col2:
    if ui.button("Reset"):
        ui.session_state.chat_log = []
        ui.rerun()

# ----------- DISPLAY CHAT HISTORY -----------
for entry in ui.session_state.chat_log:
    with ui.chat_message(entry["sender"]):
        ui.markdown(entry["message"])

# ----------- USER INPUT -----------
query = ui.chat_input("Type your question here...")

# ----------- MODEL + PIPELINE SETUP -----------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant who gives clear and concise answers."),
        ("user", "User Query: {input}")
    ]
)

model = OllamaLLM(model="llama3")   # ← fixed: llama2 changed to llama3
output_handler = StrOutputParser()

pipeline = chat_prompt | model | output_handler

# ----------- RESPONSE GENERATION -----------
if query:
    ui.session_state.chat_log.append({"sender": "user", "message": query})

    with ui.chat_message("user"):
        ui.markdown(query)

    with ui.chat_message("assistant"):
        with ui.spinner("Generating response..."):
            try:
                answer = pipeline.invoke({"input": query})
                ui.markdown(answer)
                ui.session_state.chat_log.append(
                    {"sender": "assistant", "message": answer}
                )
            except Exception as e:
                ui.error(f"Error: {str(e)}")