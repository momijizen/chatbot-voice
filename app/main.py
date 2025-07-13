# app/main.py
from flask import Flask, request, render_template, jsonify
import openai
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Load and Embed Documents ===
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


def build_faiss(docs):
    embs = [get_embedding(doc) for doc in docs]
    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs))
    return index, docs

print("\U0001F4C4 Loading docs...")
doc_texts = load_documents("docs")
print("\U0001F527 Building index...")
index, raw_docs = build_faiss(doc_texts)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    qvec = get_embedding(query)
    D, I = index.search(np.array([qvec]), k=3)
    context = "\n".join([raw_docs[i][:1000] for i in I[0]])

    prompt = f"""คุณคือผู้ช่วย AI

บริบท:
{context}

คำถาม: {query}
กรุณาตอบอย่างกระชับโดยใช้ข้อมูลข้างต้น หากไม่พบข้อมูลให้ระบุว่าไม่ทราบ
"""

    res = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    answer = res.choices[0].message.content.strip()
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
