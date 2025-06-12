import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入项目模块
from models.transformer import Transformer
from utils.data import load_data, prepare_datasets, TranslationDataset
from utils.tokenizer import Tokenizer
from utils.metrics import bleu_score
from utils.scheduler import WarmupScheduler

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, clip, save_dir, save_every, eval_every):
    """训练模型的主函数"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 训练进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]")
        
        for batch in train_iter:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            
            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                             tgt[:, 1:].contiguous().view(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 更新损失
            train_loss += loss.item()
            
            # 更新进度条
            train_iter.set_postfix({"loss": loss.item()})
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证阶段
        val_loss, val_bleu = evaluate_model(model, val_loader, criterion, device)
        
        # 计算时间
        epoch_time = time.time() - start_time
        
        # 打印训练信息
        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.4f}")
        
        # 保存模型
        if epoch % save_every == 0:
            model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f"  Model saved to {model_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  Best model updated (val_loss: {best_val_loss:.4f})")
        
        # 评估模型
        if epoch % eval_every == 0:
            print("  Evaluating model on validation set...")
            val_loss, val_bleu = evaluate_model(model, val_loader, criterion, device)
            print(f"  Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.4f}")
    
    return model

def evaluate_model(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    
    # 用于计算BLEU分数
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            output = model(src, tgt[:, :-1])
            
            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                             tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
            
            # 为BLEU分数准备数据
            # 这里简化处理，实际应用中需要根据tokenizer和数据集格式调整
            # 假设tgt是已经转换为文本的目标序列
            # 假设output可以转换为预测的文本序列
            
            # 
            # for i in range(len(tgt)):
            #     ref = tokenizer.decode(tgt[i].tolist())
            #     hyp = tokenizer.decode(torch.argmax(output[i], dim=1).tolist())
            #     references.append([ref.split()])
            #     hypotheses.append(hyp.split())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 计算BLEU分数
    # bleu = bleu_score(references, hypotheses) if references and hypotheses else 0.0
    # 简化：暂时返回0，实际应用中应该计算真实的BLEU分数
    bleu = 0.0
    
    return avg_loss, bleu

def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--config', type=str, default='config/training_config.json',
                        help='Path to training configuration file')
    parser.add_argument('--model_config', type=str, default='config/model_config.json',
                        help='Path to model configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    args = parser.parse_args()
    
    # 加载训练配置
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 加载模型配置
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    
    # 更新配置（如果命令行参数提供）
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_data, val_data, src_vocab, tgt_vocab = load_data(config['data_path'])
    
    # 准备数据集
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, 
                                       max_length=model_config['max_seq_length'])
    val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab,
                                     max_length=model_config['max_seq_length'])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])
    
    # 初始化模型
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        d_ff=model_config['d_ff'],
        max_seq_length=model_config['max_seq_length'],
        dropout=model_config['dropout']
    ).to(device)
    
    # 加载预训练模型（如果提供）
    if args.model_path:
        print(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0通常是padding的索引
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                           betas=(0.9, 0.98), eps=1e-9)
    
    # 定义学习率调度器
    scheduler = WarmupScheduler(
        optimizer, 
        d_model=model_config['d_model'], 
        warmup_steps=4000
    )
    
    # 训练模型
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        clip=config['clip'],
        save_dir=config['output_dir'],
        save_every=config['save_every'],
        eval_every=config['eval_every']
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
