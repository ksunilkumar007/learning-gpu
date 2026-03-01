# RAG — Retrieval Augmented Generation
> RHEL 9 · NVIDIA L4 · PyTorch · Ollama · ChromaDB

---

## What is RAG?

RAG gives an LLM access to your own documents at query time.
Without RAG, an LLM only knows what it was trained on.
With RAG, it can answer questions from **your** data.

```
Without RAG:
  User: "What does our refund policy say?"
  LLM:  "I don't know your company's refund policy."

With RAG:
  User: "What does our refund policy say?"
  RAG:  → searches your docs → finds refund policy chunk
  LLM:  "According to your policy, refunds are processed in 5-7 days..."
```

---

## The RAG Flow

```
Your documents (PDF, txt, md)
        ↓
01 → chunk into paragraphs
        ↓
02 → embed each chunk → float vectors
        ↓
03 → store vectors in ChromaDB
        ↓
User asks a question
        ↓
04 → embed the question
        ↓
05 → find similar chunks in ChromaDB (cosine similarity)
        ↓
06 → inject chunks into prompt → send to Ollama
        ↓
07 → return answer via FastAPI
```

---

## Environment

```
Python:    3.9
GPU:       NVIDIA L4 (23GB VRAM)
CUDA:      12.4
PyTorch:   2.6.0+cu124
Ollama:    running via Podman on port 11434
Model:     tinyllama (or any Ollama model)
```

---

## Setup

```bash
# Clone / navigate to project
cd ~/learning-gpu

# Install dependencies
uv sync

# Start Ollama (if not running)
podman start ollama

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

---

## Dependencies

```toml
[project]
dependencies = [
    "torch",
    "transformers",
    "sentence-transformers",  # better embeddings
    "chromadb",               # vector database
    "pypdf",                  # PDF loading
    "fastapi",                # REST API
    "uvicorn",                # ASGI server
    "requests",               # Ollama API calls
]
```

---

## Project Structure

```
src/rag/
├── 01_vector_basics.py       # cosine similarity from scratch
├── 02_embeddings_search.py   # sentence-transformers embeddings
├── 03_chromadb.py            # store + query with ChromaDB
├── 04_document_loader.py     # load PDF, txt, markdown
├── 05_rag_pipeline.py        # full retrieval pipeline
├── 06_rag_ollama.py          # RAG with local Ollama LLM
├── 07_rag_api.py             # wrap RAG in FastAPI
└── docs/                     # your documents go here
```

---

## Key Concepts

### Chunking
```
Document → split into overlapping paragraphs
"My name is John. I like cats. Cats are great." →
  chunk1: "My name is John. I like cats."
  chunk2: "I like cats. Cats are great."
```

### Embeddings
```
"What is overfitting?" → [0.2, 0.8, 0.1, ...]  384-dim vector
"overfitting in ML"    → [0.3, 0.7, 0.2, ...]  ← similar vector
"recipe for pasta"     → [0.9, 0.1, 0.8, ...]  ← very different
```

### Cosine Similarity
```
similarity = dot(A, B) / (|A| × |B|)
1.0  → identical
0.9  → very similar
0.5  → somewhat related
0.0  → completely different
```

### Prompt Injection
```python
prompt = f"""
Answer the question using only the context below.

Context:
{retrieved_chunks}

Question: {user_question}
Answer:
"""
```

---

## Results Summary

| Block | Script | What it does |
|-------|--------|-------------|
| 01 | `01_vector_basics.py` | Cosine similarity from scratch |
| 02 | `02_embeddings_search.py` | Embed + search documents |
| 03 | `03_chromadb.py` | Persistent vector database |
| 04 | `04_document_loader.py` | Load PDF, txt, markdown |
| 05 | `05_rag_pipeline.py` | Full retrieval pipeline |
| 06 | `06_rag_ollama.py` | End-to-end RAG with LLM |
| 07 | `07_rag_api.py` | REST API for RAG |

---

## Ollama Commands

```bash
# Start container
podman start ollama

# List models
podman exec ollama ollama list

# Pull a model
podman exec ollama ollama pull tinyllama
podman exec ollama ollama pull llama3

# Test
curl http://localhost:11434/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello", "stream": false}'
```

## Step 01 — Vector Basics

> **NOTE:** Cosine similarity is the core of every RAG system.
> It measures the angle between two vectors — closer to 1.0 means
> more similar.
>
> ```
> similarity = dot(A, B) / (|A| × |B|)
> 1.0  → identical
> 0.9  → very similar
> 0.0  → completely different
> ```

### src/rag/01_vector_basics.py

```python
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunks = {
    "refund_policy":   torch.tensor([0.9, 0.1, 0.2, 0.8, 0.1]),
    "shipping_policy": torch.tensor([0.8, 0.2, 0.1, 0.9, 0.2]),
    "return_policy":   torch.tensor([0.85, 0.15, 0.25, 0.75, 0.1]),
    "privacy_policy":  torch.tensor([0.1, 0.9, 0.8, 0.1, 0.7]),
    "cookie_policy":   torch.tensor([0.2, 0.8, 0.9, 0.2, 0.6]),
}

question = torch.tensor([0.88, 0.12, 0.22, 0.82, 0.15])

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def search(question, chunks, top_k=3):
    scores = []
    for name, vec in chunks.items():
        score = cosine_similarity(question, vec)
        scores.append((name, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

results = search(question, chunks, top_k=5)
for name, score in results:
    bar = "█" * int(score * 20)
    print(f"  {name:20s} → {score:.4f} {bar}")

print(f"\nTop match: {results[0][0]}")
```

```
Cosine similarity scores:
  refund_policy        → 0.9987 ███████████████████
  return_policy        → 0.9979 ███████████████████
  shipping_policy      → 0.9883 ███████████████████
  cookie_policy        → 0.4255 ████████
  privacy_policy       → 0.3226 ██████

Top match: refund_policy
```

> **Key insight:** ChromaDB, Pinecone, Weaviate — all vector databases
> do exactly this under the hood. The difference is they do it at scale,
> persistently, and efficiently with approximate nearest neighbor search.

---

## Step 02 — Real Embeddings with Sentence Transformers

> **NOTE:** Real RAG uses pretrained embedding models that capture meaning.
> The same model must be used for both documents and queries.
>
> ```
> Block 1: "refund_policy" → [0.9, 0.1, 0.2]     ← hand crafted
> Block 2: "refund_policy" → [0.23, -0.41, 0.87, ...] ← 384 dims, learned
> ```

### src/rag/02_embeddings_search.py

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# all-MiniLM-L6-v2 → 384 dimensions, fast, great quality
model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

chunks = [
    "Refunds are processed within 5-7 business days to the original payment method.",
    "Shipping is free on orders over $50. Standard delivery takes 3-5 business days.",
    "Items can be returned within 30 days of purchase in original condition.",
    "We collect your email and name to process orders and send shipping updates.",
    "Our website uses cookies to improve your browsing experience.",
]

# Embed all chunks
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

def search(question, chunks, chunk_embeddings, top_k=2):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = F.cosine_similarity(
        question_embedding.unsqueeze(0),
        chunk_embeddings
    )
    top_k_scores, top_k_indices = scores.topk(top_k)
    return [(chunks[idx], score.item())
            for score, idx in zip(top_k_scores, top_k_indices)]

questions = [
    "How do I get my money back?",
    "How long does delivery take?",
    "Can I send back a product?",
    "What data do you collect about me?",
]

for question in questions:
    print(f"\nQ: {question}")
    for chunk, score in search(question, chunks, chunk_embeddings):
        print(f"  [{score:.4f}] {chunk}")
```

```
Q: How do I get my money back?
  [0.4704] Refunds are processed within 5-7 business days...
  [0.3883] Items can be returned within 30 days...

Q: How long does delivery take?
  [0.5798] Shipping is free on orders over $50...
  [0.3057] We collect your email and name...

Q: Can I send back a product?
  [0.5197] Items can be returned within 30 days...
  [0.3040] We collect your email and name...

Q: What data do you collect about me?
  [0.1981] We collect your email and name...
  [0.1283] Our website uses cookies...
```

> **Key insight:** The model matched by meaning not keywords.
> "money back" → refunds, "send back" → returned, "delivery" → shipping.
> The ranking matters more than the absolute score.
>
> ```
> Embedding model  → all-MiniLM-L6-v2  → finds relevant chunks
> Generation model → tinyllama          → generates the answer
> You need both in a RAG system
> ```

---

## Step 03 — ChromaDB Persistent Vector Database

> **NOTE:** ChromaDB persists embeddings to disk so you embed once
> and search forever. No re-embedding on restart.
>
> ```
> Block 2 → in-memory  → lost on exit
> Block 3 → ChromaDB   → persists to disk, survives restarts
> ```

### src/rag/03_chromadb.py

```python
# Fix sqlite3 version for ChromaDB on RHEL 9
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="src/rag/docs/chroma_db")
collection = client.get_or_create_collection(
    name="company_policies",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    {"id": "refund_001",   "text": "Refunds are processed within 5-7 business days to the original payment method.",  "metadata": {"category": "refund",   "source": "policy.txt"}},
    {"id": "shipping_001", "text": "Shipping is free on orders over $50. Standard delivery takes 3-5 business days.", "metadata": {"category": "shipping", "source": "policy.txt"}},
    {"id": "return_001",   "text": "Items can be returned within 30 days of purchase in original condition.",          "metadata": {"category": "return",   "source": "policy.txt"}},
    {"id": "privacy_001",  "text": "We collect your email and name to process orders and send shipping updates.",       "metadata": {"category": "privacy",  "source": "policy.txt"}},
    {"id": "cookie_001",   "text": "Our website uses cookies to improve your browsing experience.",                    "metadata": {"category": "privacy",  "source": "policy.txt"}},
]

# Embed and store — only if collection is empty
if collection.count() == 0:
    embeddings = embedder.encode(
        [d["text"] for d in documents],
        convert_to_tensor=False
    ).tolist()
    collection.add(
        ids        = [d["id"] for d in documents],
        documents  = [d["text"] for d in documents],
        embeddings = embeddings,
        metadatas  = [d["metadata"] for d in documents],
    )
    print(f"Stored {collection.count()} documents")
else:
    print(f"Collection already has {collection.count()} documents")

# Search
def search(question, top_k=2):
    question_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    return list(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ))

questions = [
    "How do I get my money back?",
    "How long does delivery take?",
    "What data do you store about me?",
]

for q in questions:
    print(f"\nQ: {q}")
    for doc, dist, meta in search(q):
        print(f"  [{1-dist:.4f}] [{meta['category']}] {doc}")
```

```
First run:
  Stored 5 documents

  Q: How do I get my money back?
    [0.4704] [refund] Refunds are processed within 5-7 business days...
    [0.3883] [return] Items can be returned within 30 days...

Second run:
  Collection already has 5 documents  ← no re-embedding!
```

> **Key insight:** Embed once, search forever.
> `1 - distance` converts ChromaDB distance to similarity score.
> Metadata lets you filter by category, source, date etc.

---

## Step 04 — Document Loader

> **NOTE:** Real RAG loads from actual files. Chunking is critical —
> documents are too long to embed whole. Overlapping chunks prevent
> context being lost at boundaries.
>
> ```
> chunk_size too large → too much noise
> chunk_size too small → loses context
> overlap too small    → sentences cut at boundaries
> Sweet spot: chunk_size=200-500 words, overlap=20-50 words
> ```

### Sample documents

```bash
# Create test documents
cat > src/rag/docs/policy.txt << 'EOF'
REFUND POLICY
Refunds are processed within 5-7 business days...
SHIPPING POLICY
Shipping is free on orders over $50...
EOF

cat > src/rag/docs/faq.md << 'EOF'
# Frequently Asked Questions
## How long does shipping take?
Standard shipping takes 3-5 business days...
EOF
```

### src/rag/04_document_loader.py

```python
import os
from pypdf import PdfReader

def load_txt(filepath):
    with open(filepath, "r") as f:
        return f.read()

def load_md(filepath):
    with open(filepath, "r") as f:
        return f.read()

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text   = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_and_chunk(filepath, chunk_size=50, overlap=10):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":   text = load_txt(filepath)
    elif ext == ".md":  text = load_md(filepath)
    elif ext == ".pdf": text = load_pdf(filepath)
    else: raise ValueError(f"Unsupported: {ext}")
    chunks = chunk_text(text, chunk_size, overlap)
    return [{"text": c, "source": os.path.basename(filepath)} for c in chunks]

def load_docs_folder(folder, chunk_size=50, overlap=10):
    supported  = {".txt", ".md", ".pdf"}
    all_chunks = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            continue
        filepath = os.path.join(folder, filename)
        chunks   = load_and_chunk(filepath, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"  {filename:30s} → {len(chunks)} chunks")
    return all_chunks

all_chunks = load_docs_folder("src/rag/docs")
print(f"Total chunks: {len(all_chunks)}")
print(f"Sources: {set(c['source'] for c in all_chunks)}")
```

```
  faq.md                         → 3 chunks
  policy.txt                     → 4 chunks

Total chunks: 7
Sources: {'policy.txt', 'faq.md'}
```

> **Key insight:** Each chunk is a self-contained piece of information
> that can be retrieved and passed to an LLM. The source metadata lets
> you tell the user where the answer came from.

---

## Step 05 — Full RAG Pipeline

> **NOTE:** This is the complete RAG pipeline — load, embed, store,
> retrieve, and generate. The prompt injection pattern is the most
> important concept in RAG.
>
> ```
> docs → chunk → embed → ChromaDB → retrieve → prompt → Ollama → answer
> ```

### src/rag/05_rag_pipeline.py

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

embedder   = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text, chunk_size=30, overlap=5):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def load_docs_folder(folder):
    supported  = {".txt", ".md"}
    all_chunks = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            continue
        with open(os.path.join(folder, filename)) as f:
            text = f.read()
        chunks = chunk_text(text)
        all_chunks.extend([
            {"text": c, "source": filename, "id": f"{filename}_{i}"}
            for i, c in enumerate(chunks)
        ])
    return all_chunks

if collection.count() == 0:
    chunks     = load_docs_folder("src/rag/docs")
    embeddings = embedder.encode(
        [c["text"] for c in chunks], convert_to_tensor=False
    ).tolist()
    collection.add(
        ids        = [c["id"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        embeddings = embeddings,
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

def retrieve(question, top_k=3):
    results = collection.query(
        query_embeddings=[embedder.encode(question).tolist()],
        n_results=top_k
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))

def generate(prompt):
    return requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).json()["response"]

def rag(question):
    chunks  = retrieve(question, top_k=3)
    context = "\n".join([f"- {doc} (source: {meta['source']})"
                         for doc, meta in chunks])
    prompt  = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    answer = generate(prompt)
    return answer, chunks
```

```
Q: How do I get a refund?
A: Contact support@company.com with your order number.
Sources: {'policy.txt'}

Q: How long does shipping take?
A: Standard 3-5 days. Express 1-2 days.
Sources: {'policy.txt', 'faq.md'}

Q: Can I return a sale item?
A: No, Sale items are final sale and cannot be returned.
Sources: {'policy.txt', 'faq.md'}

Q: What payment methods do you accept?
A: Visa, Mastercard, Amex, PayPal, and Apple Pay.
Sources: {'faq.md'}
```

> **Key insight:** Model size matters for generation quality.
>
> ```
> tinyllama → retrieval works, generation confused
> llama3    → retrieval works, generation perfect
> ```
>
> The prompt injection pattern is universal:
> ```
> "Answer using only the context below.
>  If not in context, say 'I don't know'."
> ```

---

## Step 06 — RAG with Ollama LLM

> **NOTE:** Block 5 retrieved chunks. Block 6 adds the generation step —
> the LLM reads the retrieved chunks and produces a grounded answer.
>
> ```
> Block 5: question → ChromaDB → chunks
> Block 6: question → ChromaDB → chunks → Ollama → answer
> ```
>
> Built in 3 steps:
> ```
> Step 1: Verify Ollama connection
> Step 2: Add retrieval from ChromaDB
> Step 3: Wire retrieval → prompt → generation
> ```

### Step 1 — Verify Ollama connection

```python
import requests

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

response = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={
        "model":  OLLAMA_MODEL,
        "prompt": "Say hello in one sentence.",
        "stream": False,
    }
)

print(response.json()["response"])
```

```
Hello!
```

### Step 2 — Add retrieval

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f"Chunks in DB: {collection.count()}")

def retrieve(question, top_k=3):
    embedding = embedder.encode(question).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return list(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ))

question = "How do I get a refund?"
hits     = retrieve(question)

print(f"\nQ: {question}")
for doc, dist, meta in hits:
    print(f"  [{1-dist:.3f}] {doc}")
```

```
Chunks in DB: 11

Q: How do I get a refund?
  [0.513] REFUND POLICY Refunds are processed within 5-7 business days...
  [0.438] ...To request a refund contact support@company.com with your order number...
  [0.389] ...sale and cannot be returned. To start a return visit returns.company.com...
```

> **Key insight:** Overlapping chunks mean context bleeds between them.
> Notice the refund policy text appears in chunks 1 and 2 — that's the
> overlap window working as designed, preventing context from being lost
> at chunk boundaries.

### Step 3 — Add generation (full RAG loop)

### src/rag/06_rag_ollama.py

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f"Device: {device}")
print(f"Chunks in DB: {collection.count()}\n")

def retrieve(question, top_k=3):
    embedding = embedder.encode(question).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return list(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ))

def generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    return response.json()["response"].strip()

def rag(question):
    hits    = retrieve(question, top_k=3)
    context = "\n".join(
        f"- {doc}" for doc, dist, meta in hits
    )
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    answer  = generate(prompt)
    sources = list({meta["source"] for _, _, meta in hits})
    return answer, sources

questions = [
    "How do I get a refund?",
    "How long does shipping take?",
    "Can I return a sale item?",
    "What payment methods do you accept?",
]

for q in questions:
    print(f"Q: {q}")
    answer, sources = rag(q)
    print(f"A: {answer}")
    print(f"Sources: {sources}")
    print()
```

```
Device: cuda
Chunks in DB: 11

Q: How do I get a refund?
A: To request a refund, contact support@company.com with your order number.
Sources: ['policy.txt']

Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.
Sources: ['faq.md', 'policy.txt']

Q: Can I return a sale item?
A: No, sale items are final sale and cannot be returned or exchanged.
Sources: ['faq.md', 'policy.txt']

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, Amex, PayPal, and Apple Pay.
Sources: ['faq.md']
```

> **Key insight:** The LLM answers only from your documents, not from
> training memory. If the answer isn't in the retrieved chunks, it says
> "I don't know" — that's the prompt injection guardrail working.
>
> ```
> llama3 reads ONLY the 3 retrieved chunks
> not the internet, not its training data
> just what you gave it
> ```
>
> Model size matters:
> ```
> tinyllama → retrieval works, generation confused
> llama3    → retrieval works, generation perfect
> ```

---

## Step 07 — FastAPI REST API

> **NOTE:** Block 6 was a Python function call. Block 7 wraps that exact
> same logic in an HTTP API. The RAG code doesn't change — you're just
> adding a layer so anything can call it: curl, a web app, another service.
>
> ```
> Block 6: rag("question")          → answer
> Block 7: POST /query {"question"} → answer  ← same logic, HTTP interface
> ```
>
> Built in 4 steps:
> ```
> Step 1: Bare FastAPI app        → /health returns ok
> Step 2: Load RAG at startup     → /health reports device + chunks
> Step 3: Add /query endpoint     → POST question, get answer + sources
> Step 4: Add /chunks endpoint    → debug view of ChromaDB over HTTP
> ```

### Step 1 — Bare FastAPI app

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```
INFO:     Uvicorn running on http://0.0.0.0:8000

$ curl -s http://localhost:8000/health
{"status":"ok"}
```

### Step 2 — Load RAG components at startup

```python
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "chunks": collection.count(),
    }
```

```
Device: cuda
Chunks in DB: 11

$ curl -s http://localhost:8000/health
{"status":"ok","device":"cuda","chunks":11}
```

> **Key insight:** Components load once when the server starts, not on
> every request. Loading the embedding model on every query would be
> very slow.

### Step 3 — Add /query endpoint

```python
class QueryRequest(BaseModel):
    question: str
    top_k:    int = 3

@app.post("/query")
def query(req: QueryRequest):
    hits    = retrieve(req.question, req.top_k)
    context = "\n".join(f"- {doc}" for doc, dist, meta in hits)
    prompt  = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {req.question}
Answer:"""

    answer  = generate(prompt)
    sources = list({meta["source"] for _, _, meta in hits})
    return {
        "question": req.question,
        "answer":   answer,
        "sources":  sources,
    }
```

```
$ curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I get a refund?"}' \
  | python3 -m json.tool
{
    "question": "How do I get a refund?",
    "answer": "To request a refund, contact support@company.com with your
               order number. Refund requests must be submitted within 60
               days of purchase. Refunds are processed within 5-7 business
               days to the original payment method.",
    "sources": ["policy.txt"]
}
```

> **Key insight:** `BaseModel` from Pydantic validates the request
> automatically. If `question` is missing, FastAPI returns a 422 error
> before your code even runs.

### Step 4 — Add /chunks debug endpoint

```python
@app.get("/chunks")
def chunks(limit: int = 10):
    results = collection.get(limit=limit)
    return {
        "total": collection.count(),
        "shown": len(results["ids"]),
        "items": [
            {
                "id":     i,
                "source": m.get("source", "?"),
                "text":   d[:80],
            }
            for i, m, d in zip(
                results["ids"],
                results["metadatas"],
                results["documents"],
            )
        ],
    }
```

```
$ curl -s "http://localhost:8000/chunks?limit=3" | python3 -m json.tool
{
    "total": 11,
    "shown": 3,
    "items": [
        {"id": "faq.md_0",  "source": "faq.md",     "text": "# Frequently Asked Questions ..."},
        {"id": "faq.md_1",  "source": "faq.md",     "text": "a sale item? No. Sale items are final..."},
        {"id": "policy.txt_0", "source": "policy.txt", "text": "REFUND POLICY Refunds are processed..."}
    ]
}
```

### src/rag/07_rag_api.py

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path="src/rag/docs/chroma_db_rag")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f"Device: {device}")
print(f"Chunks in DB: {collection.count()}")

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str
    top_k:    int = 3

def retrieve(question, top_k=3):
    embedding = embedder.encode(question).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return list(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ))

def generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    return response.json()["response"].strip()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "chunks": collection.count(),
    }

@app.post("/query")
def query(req: QueryRequest):
    hits    = retrieve(req.question, req.top_k)
    context = "\n".join(f"- {doc}" for doc, dist, meta in hits)
    prompt  = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {req.question}
Answer:"""

    answer  = generate(prompt)
    sources = list({meta["source"] for _, _, meta in hits})
    return {
        "question": req.question,
        "answer":   answer,
        "sources":  sources,
    }

@app.get("/chunks")
def chunks(limit: int = 10):
    results = collection.get(limit=limit)
    return {
        "total": collection.count(),
        "shown": len(results["ids"]),
        "items": [
            {
                "id":     i,
                "source": m.get("source", "?"),
                "text":   d[:80],
            }
            for i, m, d in zip(
                results["ids"],
                results["metadatas"],
                results["documents"],
            )
        ],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```
# Final end-to-end test

$ curl -s http://localhost:8000/health | python3 -m json.tool
{
    "status": "ok",
    "device": "cuda",
    "chunks": 11
}

$ curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I return a sale item?", "top_k": 2}' \
  | python3 -m json.tool
{
    "question": "Can I return a sale item?",
    "answer": "Cannot be returned or exchanged.",
    "sources": ["policy.txt", "faq.md"]
}

$ curl -s "http://localhost:8000/chunks?limit=3" | python3 -m json.tool
{
    "total": 11,
    "shown": 3,
    "items": [...]
}
```

> **Key insight:** The answer to "Can I return a sale item?" pulled from
> both `policy.txt` and `faq.md` — retrieval working across multiple
> documents. The LLM distilled two chunks into one clean answer.
>
> ```
> GET  /health  → liveness check — device, chunk count
> POST /query   → question in, answer + sources out
> GET  /chunks  → debug view of what's stored in ChromaDB
> ```
>
> The full RAG system is now accessible over HTTP. Any app, service,
> or script can query your documents without knowing anything about
> embeddings, ChromaDB, or Ollama.

---