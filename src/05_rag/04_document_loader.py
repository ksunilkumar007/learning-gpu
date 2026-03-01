import os
import re

def load_txt(filepath):
    with open(filepath, "r") as f:
        return f.read()

def load_md(filepath):
    with open(filepath, "r") as f:
        return f.read()

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks by words."""
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap with next chunk

    return chunks

def load_and_chunk(filepath, chunk_size=50, overlap=10):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        text = load_txt(filepath)
    elif ext == ".md":
        text = load_md(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = chunk_text(text, chunk_size, overlap)
    return [{"text": c, "source": os.path.basename(filepath)} for c in chunks]

# Load documents
docs_path = "src/rag/docs"
all_chunks = []

for filename in ["policy.txt", "faq.md"]:
    filepath = os.path.join(docs_path, filename)
    chunks   = load_and_chunk(filepath)
    all_chunks.extend(chunks)
    print(f"Loaded {filename} → {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")
print("\nSample chunks:")
for chunk in all_chunks[:3]:
    print(f"\n  [{chunk['source']}]")
    print(f"  {chunk['text'][:120]}...")


from pypdf import PdfReader

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text   = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_and_chunk(filepath, chunk_size=50, overlap=10):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        text = load_txt(filepath)
    elif ext == ".md":
        text = load_md(filepath)
    elif ext == ".pdf":
        text = load_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = chunk_text(text, chunk_size, overlap)
    return [{"text": c, "source": os.path.basename(filepath)} for c in chunks]

def load_docs_folder(folder, chunk_size=50, overlap=10):
    """Load all supported files from a folder."""
    supported = {".txt", ".md", ".pdf"}
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

# Load all docs
print("Loading all documents:")
all_chunks = load_docs_folder("src/rag/docs")

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Sources: {set(c['source'] for c in all_chunks)}")
print(f"\nAll chunks:")
for i, chunk in enumerate(all_chunks):
    print(f"\n  [{i}] [{chunk['source']}]")
    print(f"  {chunk['text'][:100]}...")   