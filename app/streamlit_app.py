# app/streamlit_app.py
import streamlit as st
import openai
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="AI Q&A Chat", layout="centered")
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Load and Embed Documents ===
@st.cache_resource
def load_documents(folder="docs"):
    texts = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif fname.endswith(".pdf"):
            reader = PdfReader(path)
            texts.append("\n".join([p.extract_text() or '' for p in reader.pages]))
        elif fname.endswith(".docx"):
            doc = Document(path)
            texts.append("\n".join([p.text for p in doc.paragraphs]))
    return texts


def get_embedding(text):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype="float32")

@st.cache_resource
def build_faiss(docs):
    embs = [get_embedding(doc) for doc in docs]
    if not embs:
        raise ValueError("ไม่มีเอกสารสำหรับสร้าง Index")
    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs))
    return index, docs

# === Load and Prepare ===
st.markdown("## 🌐 SRT - DX Chatbot")
doc_texts = load_documents("docs")
index, raw_docs = build_faiss(doc_texts)

# === Chat State ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Chat Input ===
query = st.text_input("🔎 Type your question", key="chat_input")
if query:
    st.session_state.messages.append(("user", query))

    qvec = get_embedding(query)
    D, I = index.search(np.array([qvec]), k=3)
    context = "\n".join([raw_docs[i][:1000] for i in I[0]])

    prompt = f"""คุณคือผู้ช่วย AI ที่ฉลาดและอ้างอิงข้อมูลจากเอกสารได้
เอกสารอ้างอิง:
{context}

คำถาม: {query}

หากพบคำตอบในเอกสารข้างต้น ให้ใช้เพื่ออธิบาย
หากไม่พบ ให้ใช้ความรู้ทั่วไปของคุณในการตอบคำถามอย่างมั่นใจและถูกต้อง
"""

    res = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    answer = res.choices[0].message.content.strip()
    st.session_state.messages.append(("ai", answer))

# === Show Chat History ===
for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "user" else "ai"):
        st.markdown(msg)