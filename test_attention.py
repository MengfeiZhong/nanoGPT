import torch
import torch.nn as nn
from torch.nn import functional as F

B, T, C = 4, 8, 32
x = torch.randn(B,T,C)

# 1. Define the linear layer
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# 2. Compute the key, query, and value
k = key(x)   # (B, T, head_size)
q = query(x) # (B, T, head_size)
wei = q @ k.transpose(-2, -1) ## (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

# 3. Masking
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# 4. Compute the output
v = value(x) # (B, T, head_size)
out = wei @ v # (B, T, head_size)

print("Output shape:", out.shape)



