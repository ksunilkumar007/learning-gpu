import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

# Simple self-attention from scratch
# Every word asks: "which other words should I pay attention to?"

# Three matrices every attention head learns:
# Q (Query) -> what am I looking for?
# K (Key)   -> what do I contain?
# V (Value) -> what do I actually pass forward?

torch.manual_seed(42)

# Simulate a sentence: 4 words, each with 8-dim embedding
seq_len   = 4    # 4 words
embed_dim = 8    # 8-dimensional embeddings

x = torch.randn(seq_len, embed_dim).to(device)  # [4, 8]

print("Input shape:", x.shape)
print("Each row is one word's embedding")
print(x)

# Learn three projection matrices
head_dim = 8   # dimension of Q, K, V

W_Q = nn.Linear(embed_dim, head_dim, bias=False).to(device)
W_K = nn.Linear(embed_dim, head_dim, bias=False).to(device)
W_V = nn.Linear(embed_dim, head_dim, bias=False).to(device)

# Project input into Q, K, V
Q = W_Q(x)   # what am I looking for?
K = W_K(x)   # what do I contain?
V = W_V(x)   # what do I pass forward?

print("Q shape:", Q.shape)  # [4, 8]
print("K shape:", K.shape)  # [4, 8]
print("V shape:", V.shape)  # [4, 8]

# Compute attention scores — how much each word attends to each other
scores = torch.matmul(Q, K.T) / (head_dim ** 0.5)  # scale by sqrt(head_dim)
print("\nAttention scores shape:", scores.shape)     # [4, 4]
print("Attention scores:")
print(scores)

# Convert scores to probabilities with softmax
weights = F.softmax(scores, dim=-1)
print("Attention weights (sum to 1):")
print(weights)
print("Row sums:", weights.sum(dim=-1))  # should all be 1.0

# Weighted sum of values — the actual attention output
output = torch.matmul(weights, V)
print("\nAttention output shape:", output.shape)  # [4, 8]
print("Attention output:")
print(output)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        output  = torch.matmul(weights, V)
        return output, weights

# Test it
attention = SelfAttention(embed_dim=8).to(device)

# Add batch dimension — [batch, seq_len, embed_dim]
x_batch = x.unsqueeze(0)  # [1, 4, 8]
output, weights = attention(x_batch)

print("Input shape: ", x_batch.shape)   # [1, 4, 8]
print("Output shape:", output.shape)    # [1, 4, 8]
print("Weights shape:", weights.shape)  # [1, 4, 4]
print("\nAttention weights per word:")
for i, row in enumerate(weights.squeeze()):
    print(f"  word{i} attends: {[f'{w:.2f}' for w in row.tolist()]}")
