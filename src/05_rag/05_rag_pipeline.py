# Fix sqlite3 for ChromaDB on RHEL 9
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# 1. Load and chunk documents
def load_txt(filepath):
    with open(filepath, "r") as f:
        return f.read()

def load_md(filepath):
    with open(filepath, "r") as f:
        return f.read()

def chunk_text(text, chunk_size=50, overlap=10):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
# Change chunk size in load_docs_folder call
def load_docs_folder(folder):
    supported  = {".txt", ".md"}
    all_chunks = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            continue
        filepath = os.path.join(folder, filename)
        text     = load_txt(filepath) if ext == ".txt" else load_md(filepath)
        chunks   = chunk_text(text, chunk_size=30, overlap=5)  # smaller chunks
        all_chunks.extend([
            {"text": c, "source": filename, "id": f"{filename}_{i}"}
            for i, c in enumerate(chunks)
        ])
    return all_chunks

# 2. Embed and store in ChromaDB
print("Loading embedding model...")
embedder   = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

if collection.count() == 0:
    print("Loading and embedding documents...")
    chunks     = load_docs_folder("src/rag/docs")
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        convert_to_tensor=False
    ).tolist()
    collection.add(
        ids        = [c["id"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        embeddings = embeddings,
        metadatas  = [{"source": c["source"]} for c in chunks],
    )
    print(f"Stored {collection.count()} chunks")
else:
    print(f"Loaded {collection.count()} chunks from disk")

# 3. Retrieve relevant chunks
def retrieve(question, top_k=3):
    q_embedding = embedder.encode(question).tolist()
    results     = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0]
    ))

# 4. Generate answer with Ollama
def generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# 5. Full RAG pipeline
def rag(question):
    # Retrieve
    chunks = retrieve(question, top_k=3)
    context = "\n".join([f"- {doc} (source: {meta['source']})"
                         for doc, meta in chunks])

    # Build prompt
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # Generate
    answer = generate(prompt)
    return answer, chunks

# -- test ---
questions = [
    "How do I get a refund?",
    "How long does shipping take?",
    "Can I return a sale item?",
    "What payment methods do you accept?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer, chunks = rag(q)
    print(f"A: {answer.strip()}")
    print(f"Sources: {set(meta['source'] for _, meta in chunks)}")
    print("-" * 60)
