import torch
import numpy as np

def bleu_score(references, hypotheses):
    return bleu_score(references, hypotheses)

def calculate_bleu(preds, targets, tgt_vocab):
    # 将预测和目标从索引转换回文本
    idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
    
    # 准备BLEU计算的数据
    preds_text = []
    targets_text = []
    
    for pred, target in zip(preds, targets):
        # 去除填充标记和结束标记之后的内容
        pred_words = []
        for idx in pred:
            if idx_to_word[idx] == '<eos>':
                break
            pred_words.append(idx_to_word[idx])
        
        target_words = []
        for idx in target:
            if idx_to_word[idx] == '<eos>':
                break
            target_words.append(idx_to_word[idx])
        
        preds_text.append([pred_words])
        targets_text.append(target_words)
    
    # 计算BLEU分数
    bleu = bleu_score(preds_text, targets_text)
    return bleu

def calculate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # 创建掩码
            src_mask, tgt_mask = model.create_mask(src, tgt[:, :-1])
            
            # 前向传播
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                            tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
