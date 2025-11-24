import streamlit as st
import os
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx

# Configuració pàgina
st.set_page_config(page_title="Assistent virtual de Sergi Martín, Realtor", layout="wide")

# Sidebar amb logo i menú
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Portal IA")
menu = st.sidebar.radio("Menú", ["Consulta IA", "Pujar documents"])

# Inicialitzar Pinecone
pc = Pinecone(api_key=st.secrets["PineconeAPI"])
index_name = "documents-index"
indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in indexes:
    st.error(f"L'índex '{index_name}' no existeix. Crea'l a Pinecone abans de continuar.")
else:
    index = pc.Index(index_name)

# Model per embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Carpeta per persistència
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Funció per extreure text
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Funció per cridar OpenRouter amb rol system i debug
def get_openrouter_response(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OpenRouterAPI']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openchat/openchat-3.6",  # Model gratuït actualitzat
        "messages": [
            {"role": "system", "content": "Ets un expert en immobiliària a Catalunya. Respon sempre en català."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            json_resp = response.json()
            st.write("DEBUG:", json_resp)  # Mostra la resposta completa per diagnòstic
            return json_resp.get("choices", [{}])[0].get("message", {}).get("content", "Resposta buida")
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return "Error: Timeout en la connexió amb OpenRouter."
    except Exception as e:
        return f"Error inesperat: {str(e)}"

# Menú pujada documents
if menu == "Pujar documents":
    st.header("📂 Pujar documents")
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
                index.upsert([(uploaded_file.name, embedding, {"filename": uploaded_file.name, "content": text[:500]})])
                st.info(f"Embeddings enviats a Pinecone per {uploaded_file.name}.")
    # Mostrar llista de documents pujats
    st.subheader("📄 Documents pujats:")
    docs = os.listdir(UPLOAD_FOLDER)
    if docs:
        st.write(docs)
    else:
        st.write("Encara no hi ha documents pujats.")

# Menú consulta IA
elif menu == "Consulta IA":
    st.header("🔍 Consulta IA amb RAG")
    user_input = st.text_area("Escriu la teva pregunta:")
    if st.button("Generar resposta"):
        if user_input.strip():
            with st.spinner("Generant resposta..."):
                # Recuperar context de Pinecone
                query_embedding = embedder.encode(user_input).tolist()
                results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                context_texts = [match.metadata.get("content", "") for match in results.matches]
                context = "\n".join(context_texts)
                # Prompt reforçat
                PROMPT_CONTEXT = """
                Contesta com un expert en immobiliària a Catalunya.
                Prioritza informació dels documents si és rellevant.
                Si no hi ha informació als documents, explica-ho i dona la millor resposta possible sobre el sector immobiliari.
                """
                prompt = f"{PROMPT_CONTEXT}\nContext:\n{context}\nPregunta: {user_input}"
                resposta = get_openrouter_response(prompt)
                st.success(resposta)
        else:
