import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

# Self Attention from Block 19
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
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()

        # Attention
        self.attention = SelfAttention(embed_dim)

        # Feed forward — two linear layers with ReLU
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),  # expand
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),  # contract
        )

        # Layer normalization — normalizes across embedding dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. Attention + residual connection + norm
        x = self.norm1(x + self.attention(x))

        # 2. Feed forward + residual connection + norm
        x = self.norm2(x + self.ff(x))

        return x

# Test it
embed_dim = 8
ff_dim    = 32   # feedforward is usually 4x embed_dim

block = TransformerBlock(embed_dim, ff_dim).to(device)

x = torch.randn(1, 4, embed_dim).to(device)  # [batch, seq_len, embed_dim]
output = block(x)

print("Input shape: ", x.shape)
print("Output shape:", output.shape)
print("\nTransformer block parameters:")
for name, param in block.named_parameters():
    print(f"  {name:30s} | {param.shape}")


# Stack multiple transformer blocks — this is what GPT does
class TransformerStack(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Compare parameter counts at different depths
for num_blocks in [1, 2, 4, 8, 12]:
    stack = TransformerStack(embed_dim=8, ff_dim=32, num_blocks=num_blocks)
    params = sum(p.numel() for p in stack.parameters())
    print(f"  {num_blocks:2d} blocks → {params:6d} parameters")

# Real world comparison
print("\nReal world:")
print(f"  GPT-2 small  → 12 blocks, embed=768  → 117M params")
print(f"  GPT-3        → 96 blocks, embed=12288 → 175B params")
print(f"  LLaMA 7B     → 32 blocks, embed=4096  →   7B params")