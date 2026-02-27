import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
torch.manual_seed(42)

# Training text- we'll teach the model to generate similar text
text = """the cat sat on the mat
the dog ran fast
the cat ran on the mat
the dog sat fast
the cat and dog sat on the mat
the dog and cat ran fast"""

# Build character-level vocabulary
chars  = sorted(set(text))
vocab_size = len(chars)

# Tokenizer - char to int and back
ch2idx = {ch: i for i, ch in enumerate(chars)}
idx2ch = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [ch2idx[c] for c in s]
decode = lambda l: ''.join([idx2ch[i] for i in l])

# Encode full text
data = torch.tensor(encode(text), dtype=torch.long).to(device)

print(f"Text length:  {len(text)} characters")
print(f"Vocab size:   {vocab_size} unique characters")
print(f"Characters:   {chars}")
print(f"Data shape:   {data.shape}")
print(f"\nSample encode: 'cat' -> {encode('cat')}")
print(f"Sample decode: {encode('cat')} -> {decode(encode('cat'))}")

# -- Model ----
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Causal mask - can only attend to past tokens, not future
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_blocks, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # [B, T, embed]
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device)) # [T, embed]
        x = tok_emb + pos_emb                                            # [B, T, embed]
        for block in self.blocks:
            x = block(x)
        return self.head(x)                                              # [B, T, vocab]

# Initialize model
embed_dim   = 32
ff_dim      = 128
num_blocks  = 2
max_seq_len = 32

model = MiniGPT(vocab_size, embed_dim, ff_dim, num_blocks, max_seq_len).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"MiniGPT parameters: {params:,}")


# --- Training --- 
block_size = 16   # context window -> how many chars to look back
batch_size = 8

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size] for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training MiniGPT...")
for epoch in range(1000):
    x, y = get_batch()
    logits = model(x)                          # [B, T, vocab_size]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),           # [B*T, vocab_size]
        y.view(-1)                             # [B*T]
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# --- Generation ---
def generate(prompt, max_new_tokens=50):
    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_crop = idx[:, -block_size:]        # keep last block_size tokens
        logits   = model(idx_crop)
        logits   = logits[:, -1, :]            # last token prediction
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1) # sample next token
        idx      = torch.cat([idx, next_tok], dim=1)

    return decode(idx[0].tolist())

print("\nGenerated text:")
print(generate("the cat", max_new_tokens=60))
print(generate("the dog", max_new_tokens=60))
