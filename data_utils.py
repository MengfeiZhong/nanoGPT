import torch
import os

def load_data(file_path):
    """
    Read the data file and return the encoded tensor
    Return: train_data, val_data, vocab_size, encode, decode
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mapping from characters to integers and vice versa
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # Encoder: string to integers
    decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: integers to string

    data = torch.tensor(encode(text), dtype=torch.long)
    
    # 3. Split the data into train and validation sets
    n = int(0.9 * len(data)) # 90% train, 10% val
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, encode, decode

def get_batch(data, block_size, batch_size, device):
    """
    Return a batch of data. Train data and target data.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


