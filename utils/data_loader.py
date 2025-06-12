import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import pickle

class TextDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_seq_length):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # 将文本转换为索引序列
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab)
        
        # 添加开始和结束标记
        src_indices = [self.src_vocab['<sos>']] + src_indices + [self.src_vocab['<eos>']]
        tgt_indices = [self.tgt_vocab['<sos>']] + tgt_indices + [self.tgt_vocab['<eos>']]
        
        # 截断或填充序列
        src_indices = self.pad_or_truncate(src_indices)
        tgt_indices = self.pad_or_truncate(tgt_indices)
        
        return {
            'src': torch.tensor(src_indices),
            'tgt': torch.tensor(tgt_indices),
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
    def text_to_indices(self, text, vocab):
        return [vocab.get(token, vocab['<unk>']) for token in text.split()]
    
    def pad_or_truncate(self, indices):
        if len(indices) > self.max_seq_length:
            return indices[:self.max_seq_length]
        else:
            return indices + [0] * (self.max_seq_length - len(indices))

def get_dataloader(data_path, batch_size, max_seq_length, is_train=True):
    # 加载数据
    with open(os.path.join(data_path, 'train_data.json' if is_train else 'val_data.json'), 'r') as f:
        data = json.load(f)
    
    src_data = data['src']
    tgt_data = data['tgt']
    
    # 加载词汇表
    with open(os.path.join(data_path, 'src_vocab.pkl'), 'rb') as f:
        src_vocab = pickle.load(f)
    
    with open(os.path.join(data_path, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_vocab = pickle.load(f)
    
    # 创建数据集
    dataset = TextDataset(src_data, tgt_data, src_vocab, tgt_vocab, max_seq_length)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4
    )
    
    return dataloader, src_vocab, tgt_vocab
