import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer import Transformer
from utils.data_loader import get_dataloader
from utils.tokenizer import Tokenizer
from utils.metrics import calculate_bleu
from utils.scheduler import WarmupScheduler
from utils.utils import load_config, save_model


def evaluate_model(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0

    # 用于计算BLEU分数
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # 前向传播
            output = model(src, tgt[:, :-1])

            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                             tgt[:, 1:].contiguous().view(-1))

            total_loss += loss.item()

            # 为BLEU分数准备数据
            pred = torch.argmax(output, dim=-1)
            preds.extend(pred.cpu().tolist())
            targets.extend(tgt[:, 1:].cpu().tolist())

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)

    # 计算BLEU分数
    _, _, _, tgt_vocab = get_dataloader('data', 1, 1, is_train=False)
    bleu = calculate_bleu(preds, targets, tgt_vocab)

    return avg_loss, bleu


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, clip, save_dir, save_every, eval_every):
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
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

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
            save_model(model, optimizer, epoch, train_loss, model_path)
            print(f"  Model saved to {model_path}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            save_model(model, optimizer, epoch, train_loss, best_model_path)
            print(f"  Best model updated (val_loss: {best_val_loss:.4f})")

        # 评估模型
        if epoch % eval_every == 0:
            print("  Evaluating model on validation set...")
            val_loss, val_bleu = evaluate_model(model, val_loader, criterion, device)
            print(f"  Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.4f}")

    return model


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
    training_config = load_config(args.config)

    # 加载模型配置
    model_config = load_config(args.model_config)

    # 更新配置（如果命令行参数提供）
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        training_config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate

    # 设置设备
    device = torch.device(training_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    train_loader, src_vocab, tgt_vocab = get_dataloader(training_config['data_path'],
                                                        training_config['batch_size'],
                                                        model_config['max_seq_length'],
                                                        is_train=True)
    val_loader, _, _ = get_dataloader(training_config['data_path'],
                                      training_config['batch_size'],
                                      model_config['max_seq_length'],
                                      is_train=False)

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
        model, _, _, _ = save_model(model, None, args.model_path)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0通常是padding的索引
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'],
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
        num_epochs=training_config['num_epochs'],
        device=device,
        clip=training_config['clip'],
        save_dir=training_config['output_dir'],
        save_every=training_config['save_every'],
        eval_every=training_config['eval_every']
    )

    print("Training completed!")


if __name__ == "__main__":
    main()