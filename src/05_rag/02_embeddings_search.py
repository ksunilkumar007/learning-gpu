from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a small but powerful embedding model
# all-MiniLM-L6-v2 → 384 dimensions, fast, great quality
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

# Document chunks — real text this time
chunks = [
    "Refunds are processed within 5-7 business days to the original payment method.",
    "Shipping is free on orders over $50. Standard delivery takes 3-5 business days.",
    "Items can be returned within 30 days of purchase in original condition.",
    "We collect your email and name to process orders and send shipping updates.",
    "Our website uses cookies to improve your browsing experience.",
]

# Embed all chunks
print("Embedding document chunks...")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

print(f"\nChunk embeddings shape: {chunk_embeddings.shape}")
print(f"Each chunk → {chunk_embeddings.shape[1]}-dimensional vector")
print(f"Device: {chunk_embeddings.device}")

def search(question, chunks, chunk_embeddings, top_k=3):
    # Embed the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Cosine similarity between question and all chunks
    scores = F.cosine_similarity(
        question_embedding.unsqueeze(0),
        chunk_embeddings
    )

    # Get top-k results
    top_k_scores, top_k_indices = scores.topk(top_k)

    results = []
    for score, idx in zip(top_k_scores, top_k_indices):
        results.append((chunks[idx], score.item()))
    return results

# Test with different questions
questions = [
    "How do I get my money back?",
    "How long does delivery take?",
    "Can I send back a product?",
    "What data do you collect about me?",
]

for question in questions:
    print(f"\nQ: {question}")
    results = search(question, chunks, chunk_embeddings, top_k=2)
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk}")
