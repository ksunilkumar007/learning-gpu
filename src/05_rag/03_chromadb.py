# Fix sqlite3 version for ChromaDB on RHEL 9
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

# ChromaDB - persistent storage
client = chromadb.PersistentClient(path="src/rag/docs/chroma_db")

# Create or load a collection (like a table in a database)
collection = client.get_or_create_collection(
    name="company_policies",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

print(f"Collection: {collection.name}")
print(f"Documents stored: {collection.count()}")

# Document chunks with metadata
documents = [
    {
        "id": "refund_001",
        "text": "Refunds are processed within 5-7 business days to the original payment method.",
        "metadata": {"category": "refund", "source": "policy.txt"}
    },
    {
        "id": "shipping_001",
        "text": "Shipping is free on orders over $50. Standard delivery takes 3-5 business days.",
        "metadata": {"category": "shipping", "source": "policy.txt"}
    },
    {
        "id": "return_001",
        "text": "Items can be returned within 30 days of purchase in original condition.",
        "metadata": {"category": "return", "source": "policy.txt"}
    },
    {
        "id": "privacy_001",
        "text": "We collect your email and name to process orders and send shipping updates.",
        "metadata": {"category": "privacy", "source": "policy.txt"}
    },
    {
        "id": "cookie_001",
        "text": "Our website uses cookies to improve your browsing experience.",
        "metadata": {"category": "privacy", "source": "policy.txt"}
    },
]

# Embed and store - only if collection is empty
if collection.count() == 0:
    print("Storing documents in ChromaDB...")
    embeddings = embedder.encode(
        [d["text"] for d in documents],
        convert_to_tensor=False   # ChromaDB needs plain lists
    ).tolist()

    collection.add(
        ids       = [d["id"] for d in documents],
        documents = [d["text"] for d in documents],
        embeddings= embeddings,
        metadatas = [d["metadata"] for d in documents],
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

print("\nSearching ChromaDB:")
questions = [
    "How do I get my money back?",
    "How long does delivery take?",
    "What data do you store about me?",
]

for q in questions:
    print(f"\nQ: {q}")
    for doc, dist, meta in search(q):
        print(f"  [{1-dist:.4f}] [{meta['category']}] {doc}")