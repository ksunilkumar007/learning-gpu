import torch
import torch.nn as nn

device = torch.device("cuda")

# Vocabulary — maps words to integers
vocab = {
    "<pad>": 0,   # padding token
    "<unk>": 1,   # unknown word
    "the":   2,
    "cat":   3,
    "dog":   4,
    "sat":   5,
    "on":    6,
    "mat":   7,
    "ran":   8,
    "fast":  9,
}

vocab_size = len(vocab)   # 10 words
embed_dim  = 4            # each word becomes a 4-number vector

# Embedding layer — lookup table of shape [vocab_size, embed_dim]
embedding = nn.Embedding(vocab_size, embed_dim).to(device)

print("Embedding table shape:", embedding.weight.shape)
print("Vocab size:", vocab_size)
print("Embedding dimension:", embed_dim)

# Convert a sentence to integers
sentence = ["the", "cat", "sat", "on", "the", "mat"]
indices = torch.tensor([vocab[w] for w in sentence]).to(device)

print("\nSentence:", sentence)
print("Indices: ", indices)

# Look up embeddings
vectors = embedding(indices)
print("\nEmbedding vectors:")
print(vectors)
print("Shape:", vectors.shape)  # [6 words, 4 dimensions]

# Train embeddings to understand similarity
# Task: predict if two words are related (1) or not (0)
pairs = [
    ("cat", "dog",  1),   # related — both animals
    ("cat", "mat",  0),   # not related
    ("dog", "fast", 0),   # not related
    ("sat", "ran",  1),   # related — both verbs
    ("the", "on",   1),   # related — both function words
    ("cat", "ran",  0),   # not related
]

# Simple similarity model
class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, w1, w2):
        e1 = self.embedding(w1)
        e2 = self.embedding(w2)
        x  = torch.cat([e1, e2], dim=-1)  # concat both vectors
        return self.sigmoid(self.fc(x)).squeeze()

model = SimilarityModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Prepare data
w1s = torch.tensor([vocab[p[0]] for p in pairs]).to(device)
w2s = torch.tensor([vocab[p[1]] for p in pairs]).to(device)
ys  = torch.tensor([p[2] for p in pairs], dtype=torch.float).to(device)

# Train
for epoch in range(500):
    pred = model(w1s, w2s)
    loss = loss_fn(pred, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")



# Check what the model learned
model.eval()
with torch.no_grad():
    print("\nLearned embedding vectors:")
    for word, idx in vocab.items():
        if word in ["cat", "dog", "sat", "ran", "the", "on", "mat"]:
            vec = model.embedding(torch.tensor([idx]).to(device))
            print(f"  {word:6s} → {vec.squeeze().tolist()}")

    # Test similarity predictions
    print("\nSimilarity predictions:")
    test_pairs = [
        ("cat", "dog"),   # should be similar → close to 1
        ("sat", "ran"),   # should be similar → close to 1
        ("cat", "mat"),   # should be different → close to 0
        ("dog", "fast"),  # should be different → close to 0
    ]

    for w1, w2 in test_pairs:
        t1 = torch.tensor([vocab[w1]]).to(device)
        t2 = torch.tensor([vocab[w2]]).to(device)
        score = model(t1, t2).item()
        print(f"  {w1:6s} + {w2:6s} → similarity: {score:.4f}")        