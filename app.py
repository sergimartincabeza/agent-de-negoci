import streamlit as st
import os
import pinecone
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx

# Configuració pàgina
st.set_page_config(page_title="Portal IA Corporatiu", layout="wide")

# Sidebar amb logo i menú
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Portal IA")
menu = st.sidebar.radio("Menú", ["Consulta IA", "Pujar documents"])

# Inicialitzar Pinecone
pinecone.init(api_key=st.secrets["PineconeAPI"], environment="us-east-1-aws")
index = pinecone.Index("documents-index-5l8n38g")

# Model per embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Carpeta per persistència
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Funció per extreure text
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return ""
    .join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return ""
    .join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Menú pujada documents
if menu == "Pujar documents":
    st.header("F4C2 Pujar documents")
    uploaded_files = st.file_uploader("Selecciona arxius", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Arxiu {uploaded_file.name} pujat correctament.")

            # Extreure text i generar embeddings
            text = extract_text(file_path)
            if text.strip():
                embedding = embedder.encode(text).tolist()
                index.upsert([(uploaded_file.name, embedding, {"filename": uploaded_file.name})])
                st.info(f"Embeddings enviats a Pinecone per {uploaded_file.name}.")

# Menú consulta IA
elif menu == "Consulta IA":
    st.header("F50D Consulta IA amb RAG")
    user_input = st.text_area("Escriu la teva pregunta:")
    if st.button("Generar resposta"):
        if user_input.strip():
            # Recuperar context de Pinecone
            query_embedding = embedder.encode(user_input).tolist()
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            context = "
".join([match.metadata.get("filename", "") for match in results.matches])

            # Model Flan-T5
            qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
            prompt = f"Context: {context}
Pregunta: {user_input}"
            resposta = qa_pipeline(prompt)[0]['generated_text']
            st.success(resposta)
        else:
            st.warning("Introdueix una pregunta abans de continuar.")
