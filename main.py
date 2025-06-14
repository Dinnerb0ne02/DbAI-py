# DbAI-py A trandformer agent developed with pytorch and flask
# Copyright (C) Dinnerb0ne<tomma_2022@outlook.com>

import os
import sys
import subprocess
import json
import argparse
import torch
import torch.nn.functional as F
import math
import pickle
from typing import Dict, Any, Optional
from models.transformer import Transformer
from utils.utils import load_config, load_model, get_device

class ProjectManager:
    """项目管理类，负责协调不同模块的运行"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载配置
        self.training_config = self._load_config('training_config.json')
        self.model_config = self._load_config('model_config.json')
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = os.path.join(self.config_dir, config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
            return {}
    
    def _save_config(self, config: Dict[str, Any], config_file: str) -> None:
        """保存配置文件"""
        config_path = os.path.join(self.config_dir, config_file)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def train(self, **kwargs) -> None:
        """训练模型"""
        # 更新训练配置
        for key, value in kwargs.items():
            if key in self.training_config:
                self.training_config[key] = value
        
        # 保存更新后的配置
        self._save_config(self.training_config, 'training_config.json')
        
        # 构建命令
        cmd = [sys.executable, 'train.py']
        
        # 添加命令行参数
        for key, value in kwargs.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))
        
        # 执行命令
        print("开始训练模型...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"训练过程出错: {e}")
            sys.exit(1)
    
    def inference(self, **kwargs) -> None:
        """使用模型进行推理"""
        # 构建命令
        cmd = [sys.executable, 'inference.py']
        
        # 添加命令行参数
        for key, value in kwargs.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))
        
        # 执行命令
        print("开始推理...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"推理过程出错: {e}")
            sys.exit(1)
    
    def deploy(self, **kwargs) -> None:
        """部署模型为API服务"""
        # 构建命令
        cmd = [sys.executable, 'deploy.py']
        
        # 添加命令行参数
        for key, value in kwargs.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))
        
        # 执行命令
        print("部署模型为API服务...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"部署过程出错: {e}")
            sys.exit(1)
    
    def create_config(self, config_type: str, **kwargs) -> None:
        """创建或更新配置文件"""
        if config_type == 'training':
            config = self.training_config
            config_file = 'training_config.json'
        elif config_type == 'model':
            config = self.model_config
            config_file = 'model_config.json'
        else:
            print(f"错误: 未知的配置类型 '{config_type}'")
            return
        
        # 更新配置
        for key, value in kwargs.items():
            config[key] = value
        
        # 保存配置
        self._save_config(config, config_file)
        print(f"已更新 {config_type} 配置文件")
    
    def run_script(self, script_name: str, **kwargs) -> None:
        """运行自定义脚本"""
        script_path = os.path.join(self.base_dir, script_name)
        if not os.path.exists(script_path):
            print(f"错误: 脚本 '{script_name}' 不存在")
            return
        
        # 构建命令
        cmd = [sys.executable, script_name]
        
        # 添加命令行参数
        for key, value in kwargs.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))
        
        # 执行命令
        print(f"运行脚本 '{script_name}'...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"脚本运行出错: {e}")
            sys.exit(1)
    
    def show_config(self, config_type: str) -> None:
        """显示配置文件内容"""
        if config_type == 'training':
            config = self.training_config
        elif config_type == 'model':
            config = self.model_config
        else:
            print(f"错误: 未知的配置类型 '{config_type}'")
            return
        
        print(f"{config_type} 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def cli_dialogue(self, **kwargs):
        """命令行对话功能"""
        model_path = kwargs.get('model_path', 'output/best_model.pth')
        config_path = kwargs.get('config_path', 'config/training_config.json')
        model_config_path = kwargs.get('model_config_path', 'config/model_config.json')

        # 加载配置
        config = load_config(config_path)
        model_config = load_config(model_config_path)

        # 设置设备
        device = get_device()
        print(f"Using device: {device}")

        # 加载词汇表
        with open(os.path.join(config['data_path'], 'src_vocab.pkl'), 'rb') as f:
            src_vocab = pickle.load(f)

        with open(os.path.join(config['data_path'], 'tgt_vocab.pkl'), 'rb') as f:
            tgt_vocab = pickle.load(f)

        # 创建反向词汇表
        idx_to_word = {idx: word for word, idx in tgt_vocab.items()}

        # 初始化模型
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

        # 加载模型权重
        model, _, _, _ = load_model(model, None, model_path)
        model.eval()

        print("开始对话，请输入文本（输入 'quit' 退出）：")
        while True:
            text = input("> ")
            if text.lower() == 'quit':
                break

            # 将文本转换为索引序列
            src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in text.split()]
            src_indices = [src_vocab['<sos>']] + src_indices + [src_vocab['<eos>']]

            # 截断或填充序列
            if len(src_indices) > model_config['max_seq_length']:
                src_indices = src_indices[:model_config['max_seq_length']]
            else:
                src_indices = src_indices + [0] * (model_config['max_seq_length'] - len(src_indices))

            # 转换为张量
            src_tensor = torch.tensor([src_indices]).to(device)

            # 预测
            with torch.no_grad():
                output = self.greedy_decode(model, src_tensor, model_config['max_seq_length'],
                                            tgt_vocab['<sos>'], tgt_vocab['<eos>'], device)

            # 将预测结果转换回文本
            pred_indices = output[0].cpu().numpy()
            pred_words = []
            for idx in pred_indices:
                if idx == tgt_vocab['<eos>']:
                    break
                pred_words.append(idx_to_word[idx])

            pred_text = ' '.join(pred_words[1:])  # 跳过<sos>标记

            print(f"回复: {pred_text}")

    def greedy_decode(self, model, src, max_len, sos_idx, eos_idx, device):
        # 只使用编码器编码源序列
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        enc_output = model.encoder_embedding(src) * math.sqrt(model.d_model)
        enc_output = model.positional_encoding(enc_output)

        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # 初始化输出序列
        ys = torch.ones(src.size(0), 1).fill_(sos_idx).long().to(device)

        # 逐个生成标记
        for i in range(max_len - 1):
            # 创建目标序列掩码
            tgt_mask = (ys != 0).unsqueeze(1).unsqueeze(3).to(device)
            nopeak_mask = (1 - torch.triu(
                torch.ones(1, ys.size(1), ys.size(1)), diagonal=1)).bool().to(device)
            tgt_mask = tgt_mask & nopeak_mask

            # 解码
            dec_output = model.decoder_embedding(ys) * math.sqrt(model.d_model)
            dec_output = model.positional_encoding(dec_output)

            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            # 预测下一个标记
            output = model.fc_out(dec_output[:, -1])
            prob = F.softmax(output, dim=1)
            next_word = torch.argmax(prob, dim=1).unsqueeze(1)

            # 添加到输出序列
            ys = torch.cat([ys, next_word], dim=1)

            # 如果所有序列都生成了结束标记，则停止
            if (next_word == eos_idx).all():
                break

        return ys

def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='Transformer 模型项目管理工具')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--batch_size', type=int, help='批次大小')
    train_parser.add_argument('--num_epochs', type=int, help='训练轮数')
    train_parser.add_argument('--learning_rate', type=float, help='学习率')
    train_parser.add_argument('--model_path', type=str, help='预训练模型路径')
    
    # 推理命令
    inference_parser = subparsers.add_parser('inference', help='使用模型进行推理')
    inference_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    inference_parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    inference_parser.add_argument('--output_file', type=str, help='输出文件路径')
    
    # 部署命令
    deploy_parser = subparsers.add_parser('deploy', help='部署模型为API服务')
    deploy_parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    deploy_parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机')
    deploy_parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_parser.add_argument('--type', type=str, required=True, choices=['training', 'model'], help='配置类型')
    config_parser.add_argument('--show', action='store_true', help='显示配置')
    
    # 脚本命令
    script_parser = subparsers.add_parser('script', help='运行自定义脚本')
    script_parser.add_argument('--name', type=str, required=True, help='脚本名称')

    # 命令行对话命令
    cli_parser = subparsers.add_parser('cli', help='命令行对话')
    cli_parser.add_argument('--model_path', type=str, default='output/best_model.pth', help='模型路径')
    cli_parser.add_argument('--config_path', type=str, default='config/training_config.json', help='训练配置文件路径')
    cli_parser.add_argument('--model_config_path', type=str, default='config/model_config.json', help='模型配置文件路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建项目管理器
    manager = ProjectManager()
    
    # 执行相应命令
    if args.command == 'train':
        # 过滤掉None值
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'command'}
        manager.train(**kwargs)
    
    elif args.command == 'inference':
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'command'}
        manager.inference(**kwargs)
    
    elif args.command == 'deploy':
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'command'}
        manager.deploy(**kwargs)
    
    elif args.command == 'config':
        if args.show:
            manager.show_config(args.type)
        else:
            # 交互式配置
            print(f"配置 {args.type} 参数 (按Enter跳过):")
            config = manager.training_config if args.type == 'training' else manager.model_config
            
            kwargs = {}
            for key in config:
                value = input(f"{key} [{config[key]}]: ")
                if value:
                    # 尝试转换为适当的类型
                    try:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # 保持为字符串
                    
                    kwargs[key] = value
            
            manager.create_config(args.type, **kwargs)
    
    elif args.command == 'script':
        print(f"运行脚本 {args.name} (按Enter继续, Ctrl+C取消):")
        input()
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'command' and k != 'name'}
        manager.run_script(args.name, **kwargs)

    elif args.command == 'cli':
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'command'}
        manager.cli_dialogue(**kwargs)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()