
import streamlit as st
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests

# Configuració pàgina
st.set_page_config(page_title="Portal IA Corporatiu", layout="wide")

# Sidebar amb logo i FAQs
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("FAQs")
st.sidebar.write("- Com pujar documents?\n- Com funciona el RAG?\n- Contacta amb IT si tens dubtes.")
if st.sidebar.button("Mode manteniment"):
    st.warning("El portal està en mode manteniment.")

# Títol principal
st.title("Portal IA Corporatiu amb RAG i Mistral-7B via OpenRouter")

# Carpeta per persistència
DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)
INDEX_FILE = DATA_PATH / "faiss_index.pkl"
DOCS_FILE = DATA_PATH / "documents.pkl"

# Carregar model embeddings
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Carregar index i documents
if INDEX_FILE.exists() and DOCS_FILE.exists():
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    with open(DOCS_FILE, "rb") as f:
        documents = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)
    documents = []

# Pujar documents
uploaded_files = st.file_uploader("Puja documents (TXT, PDF, DOCX)", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode(errors="ignore")
        documents.append(content)
        embedding = embedder.encode([content])
        index.add(np.array(embedding, dtype="float32"))
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)
    st.success("Documents indexats correctament!")

# Historial
if "history" not in st.session_state:
    st.session_state["history"] = []

# Funció per cridar OpenRouter API amb Mistral-7B
OPENROUTER_API_KEY = st.secrets.get("OpenRouterAPI")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_answer(context, question):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-7b-instruct-v0.1",
        "messages": [
            {"role": "system", "content": "Ets un assistent corporatiu que respon amb precisió."},
            {"role": "user", "content": f"Context: {context}\nPregunta: {question}\nResposta:"}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return f"Error API: {response.status_code} - {response.text}"

# Consulta
query = st.text_input("Escriu la teva consulta:")
if st.button("Generar resposta") and query:
    try:
        q_emb = embedder.encode([query])
        D, I = index.search(np.array(q_emb, dtype="float32"), k=3)
        context = " ".join([documents[i] for i in I[0]])
        answer = generate_answer(context, query)
        st.write("**Resposta:**", answer)
        st.session_state["history"].append((query, answer))
    except Exception as e:
        st.error(f"Error en generar la resposta: {e}")

# Mostrar historial
if st.session_state["history"]:
    st.subheader("Historial de consultes")
    for q, a in st.session_state["history"]:
        st.write(f"**{q}** → {a}")
