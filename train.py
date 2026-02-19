import torch
import time
from model import BigramLanguageModel
from data_utils import load_data, get_batch

# ----------------------------------------------------------
# 1. Hyperparameters
# ----------------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 1500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
print(f"正在使用的设备: {device}")

# ----------------------------------------------------------
# 2. Load the data
# ----------------------------------------------------------
try:
    train_data, val_data, vocab_size, encode, decode = load_data('NanoGPT/input.txt')
    print(f"数据加载成功。训练集大小: {len(train_data)}，验证集大小: {len(val_data)}，词汇表大小: {vocab_size}")
except FileNotFoundError as e:
    print(e)
    exit()
# ----------------------------------------------------------
# 3. Auxiliary Functions: Evaluate the Loss
# ----------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# ----------------------------------------------------------
# 4. Initialize the Model
# ----------------------------------------------------------
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ----------------------------------------------------------
# 5. Training Loop
# ----------------------------------------------------------
print("开始训练...")
start_time = time.time()

# 只需要一层干净的循环！
for iter in range(max_iters):
    
    # 1. 定期评估和存档 (注意缩进，只有满足 if 条件才会执行这里的代码)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"迭代 {iter:4d}/{max_iters-1}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")
        
        # 存档代码必须缩进在 if 内部！
        torch.save(model.state_dict(), 'model_gpt_split.pth')
        print(f"--> [自动存档] 模型已安全保存至 model_gpt_split.pth")

    # 2. 获取数据 (注意参数顺序: 先 batch_size, 后 block_size)
    xb, yb = get_batch(train_data, batch_size, block_size, device)

    # 3. 训练和更新参数
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"训练完成。耗时: {time.time() - start_time:.2f}s")

# ----------------------------------------------------------
# 6. Save the Model
# ----------------------------------------------------------
torch.save(model.state_dict(), 'model_NanoGPT.pth')
print("模型参数已保存至 model_NanoGPT.pth")