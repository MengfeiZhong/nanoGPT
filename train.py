import torch
import os
import sys
# 将当前脚本所在目录加入搜索路径，解决导入问题
sys.path.append(os.path.dirname(__file__))

from model import BigramLanguageModel
from data_utils import load_data, get_batch

# --- 超参数 ---
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): device = 'mps'

# --- 加载数据 ---
# 使用绝对路径加载数据，解决文件找不到的问题
curr_dir = os.path.dirname(__file__)
data_path = os.path.join(curr_dir, 'input.txt')
data, vocab_size, encode, decode = load_data(data_path)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# --- 初始化模型 ---
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 训练循环 ---
for iter in range(max_iters):
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 500 == 0:
        print(f"迭代次数 {iter}, Loss: {loss.item():.4f}")

# --- 生成测试 ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n训练后生成结果：")
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))