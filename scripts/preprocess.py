import os
import json
import pickle
from collections import Counter

# 假设原始数据文件为 data/raw/your_raw_data.txt，每行包含一个源语言和目标语言的文本对，用制表符分隔
raw_data_path = 'data/raw/your_raw_data.txt'

# 读取原始数据
src_texts = []
tgt_texts = []
with open(raw_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        src, tgt = line.strip().split('\t')
        src_texts.append(src)
        tgt_texts.append(tgt)

# 创建词汇表
def create_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)
    vocab = {'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}
    idx = 4
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab

src_vocab = create_vocab(src_texts)
tgt_vocab = create_vocab(tgt_texts)

# 划分训练集和验证集
train_size = int(len(src_texts) * 0.8)
train_src = src_texts[:train_size]
train_tgt = tgt_texts[:train_size]
val_src = src_texts[train_size:]
val_tgt = tgt_texts[train_size:]

# 保存词汇表和数据
processed_dir = 'data/processed'
os.makedirs(processed_dir, exist_ok=True)

with open(os.path.join(processed_dir, 'src_vocab.pkl'), 'wb') as f:
    pickle.dump(src_vocab, f)

with open(os.path.join(processed_dir, 'tgt_vocab.pkl'), 'wb') as f:
    pickle.dump(tgt_vocab, f)

train_data = {'src': train_src, 'tgt': train_tgt}
with open(os.path.join(processed_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

val_data = {'src': val_src, 'tgt': val_tgt}
with open(os.path.join(processed_dir, 'val_data.json'), 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print("数据预处理完成，文件已保存到 data/processed 文件夹中。")