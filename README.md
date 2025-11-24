# Portal IA Corporatiu amb RAG i Pinecone

Aquesta aplicació Streamlit permet:
- Pujar documents (PDF, DOCX, TXT) i indexar-los a Pinecone.
- Consultar la IA amb context dels documents.

## Configuració
1. Pujar el projecte a GitHub.
2. Afegir la API Key de Pinecone a **Streamlit secrets**:
```
PineconeAPI = "la_teva_api_key"
```
3. Connectar el repositori a [Streamlit Cloud](https://share.streamlit.io).
