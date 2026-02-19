import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Define the Q, K, V linear layer
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Define the mask to ensure causality
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B, T, C = x.shape
        # 1. Compute the key, query, and value
        k = self.key(x)
        q = self.query(x)
        # 2. Compute the attention weights
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        # 2. Masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 3. Apply softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # 4. Weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) 
        for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A transformer block: Communication followed by Computation"""
    def __init__(self, n_embd, num_heads, block_size, dropout):
        # n_embd: embedding dimension 嵌入维度
        # num_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embd // num_heads
        # Communication layer: Multi-head self-attention
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        # Computation layer: Feed-forward network
        self.ffwd = FeedForward(n_embd, dropout)    
        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        # Communication layer
        x = x + self.sa(self.ln1(x))
        # Computation layer
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 1. Token embedding
        tok_emb = self.token_embedding_table(idx)
        # 2. Position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        # 3. Add the two embeddings
        x = tok_emb + pos_emb
        # 4. Pass through the transformer blocks
        x = self.blocks(x)
        # 5. Layer normalization
        x = self.ln_f(x)
        # 6. Linear layer to get logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
