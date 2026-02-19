import torch
from model import BigramLanguageModel
from data_utils import load_data

# 自动识别你的 M1 芯片
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 1. 获取字典和解码工具（模型本身只懂数字，我们需要 decode 把它翻译回英文）
_, _, vocab_size, encode, decode = load_data('NanoGPT/input.txt')

# 2. 搭建空壳工厂（注意：这里的参数必须和你训练时设置得一模一样！）
model = BigramLanguageModel(
    vocab_size=vocab_size, 
    n_embd=384, 
    n_head=6, 
    n_layer=6, 
    block_size=256, 
    dropout=0.2 
)

# 3. 注入灵魂！加载你千辛万苦炼出来的模型权重
print("正在唤醒模型...")
model.load_state_dict(torch.load('model_gpt_split.pth', map_location=device))
model.to(device)

# ⚠️ 关键一步：开启评估模式！这会关闭 Dropout，让模型全心全意生成，不掉链子
model.eval() 

# 4. 见证奇迹
print("\n========== 莎士比亚之魂正在苏醒 ==========\n")

# 给模型一个起始提示（比如这里给一个换行符，对应的数字通常是 0）
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# 让模型一口气写 1000 个字符
generated_idx = model.generate(context, max_new_tokens=1000)
generated_text = decode(generated_idx[0].tolist())

print(generated_text)
print("\n==========================================\n")