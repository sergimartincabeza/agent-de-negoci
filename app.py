
import streamlit as st
from langchain_openai import OpenAI  # Importació correcta
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Configuració de la pàgina
st.set_page_config(page_title="IA Premium", page_icon="🤖", layout="wide")

# Sidebar amb logo
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Menú")
st.sidebar.info("Carrega documents per millorar les respostes")

# Zona principal
st.title("Assistència IA amb RAG")

# Pujar documents
uploaded_files = st.sidebar.file_uploader("Puja documents (PDF, Word, TXT)", accept_multiple_files=True)
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} documents carregats")

# Camp de consulta
query = st.text_input("Escriu la teva pregunta:")
if st.button("Consulta IA"):
    if query.strip() == "":
        st.warning("Introdueix una pregunta abans de consultar.")
    else:
        st.info("Funcionalitat RAG en desenvolupament: resposta simulada.")
        st.write(f"Resposta per a: {query}")
