import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import pyttsx3

Checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(Checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(Checkpoint, device_map="auto", torch_dtype=torch.float32)

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=1000,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64, {base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

def text_to_speech(text, voice_id='english-us'):
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.say(text)
    engine.runAndWait()

st.set_page_config(layout='wide')

def main():
    st.title('Hi Welcome to PDF Summarizer')
    uploaded_file = st.file_uploader("Upload your PDF File", type='pdf')

    session_state = st.session_state
    if 'filepath' not in session_state:
        session_state.filepath = None

    if uploaded_file is not None:
        if st.button("Summarize"):
            st.markdown("[AR](https://sreevishnubn.github.io/AR_PROJECT123us3zmk/)")
            session_state.filepath = "data/" + uploaded_file.name
            with open(session_state.filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

    if session_state.filepath is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info("Uploaded PDF File")
            pdf_viewer = displayPDF(session_state.filepath)
        with col2:
            st.info("Summarization is belows")

            
            summary = llm_pipeline(session_state.filepath)
            st.success(summary)
            text_to_speech(summary, voice_id='english-us')
                

if __name__ == '__main__':
    main()
