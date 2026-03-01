import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Simulate document chunks as vectors
# In real RAG these come from an embedding model
# Here we manually create them to understand the concept

chunks = {
    "refund_policy":    torch.tensor([0.9, 0.1, 0.2, 0.8, 0.1]),
    "shipping_policy":  torch.tensor([0.8, 0.2, 0.1, 0.9, 0.2]),
    "return_policy":    torch.tensor([0.85, 0.15, 0.25, 0.75, 0.1]),
    "privacy_policy":   torch.tensor([0.1, 0.9, 0.8, 0.1, 0.7]),
    "cookie_policy":    torch.tensor([0.2, 0.8, 0.9, 0.2, 0.6]),
}

# User question vector
question = torch.tensor([0.88, 0.12, 0.22, 0.82, 0.15])

print("\nDocument chunks:")
for name, vec in chunks.items():
    print(f"  {name:20s} -> {vec.tolist()}")

print(f"\nQuestion vector: {question.tolist()}")

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def search(question, chunks, top_k=3):
    scores = []
    for name, vec in chunks.items():
        score = cosine_similarity(question, vec)
        scores.append((name, score))

    # Sort by similarity - highest first
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# Search
print("\nCosine similarity scores:")
results = search(question, chunks, top_k=5)
for name, score in results:
    bar = "█" * int(score * 20)
    print(f"  {name:20s}  -> {score:.4f} {bar}")

print(f"\nTop match: {results[0][0]}")
print(f"This is what RAG would retrieve for the question!")