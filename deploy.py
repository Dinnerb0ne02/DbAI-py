import torch
import torch.nn.functional as F
import json
import pickle
import os
import argparse
from flask import Flask, request, jsonify
import logging

from models.transformer import Transformer
from utils.utils import load_config, load_model, get_device

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
model = None
src_vocab = None
tgt_vocab = None
idx_to_word = None
device = None
model_config = None

def init_model(model_path, config_path, model_config_path):
    global model, src_vocab, tgt_vocab, idx_to_word, device, model_config
    
    # 加载配置
    config = load_config(config_path)
    model_config = load_config(model_config_path)
    
    # 设置设备
    device = get_device()
    logger.info(f"Using device: {device}")
    
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
    
    logger.info("Model initialized successfully")

def predict(text):
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
        output = greedy_decode(model, src_tensor, model_config['max_seq_length'], 
                              tgt_vocab['<sos>'], tgt_vocab['<eos>'], device)
    
    # 将预测结果转换回文本
    pred_indices = output[0].cpu().numpy()
    pred_words = []
    for idx in pred_indices:
        if idx == tgt_vocab['<eos>']:
            break
        pred_words.append(idx_to_word[idx])
    
    pred_text = ' '.join(pred_words[1:])  # 跳过<sos>标记
    
    return pred_text

def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
    # 只使用编码器编码源序列
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    enc_output = model.encoder_embedding(src) * math.sqrt(model.d_model)
    enc_output = model.positional_encoding(enc_output)
    
    for enc_layer in model.encoder_layers:
        enc_output = enc_layer(enc_output, src_mask)
    
    # 初始化输出序列
    ys = torch.ones(src.size(0), 1).fill_(sos_idx).long().to(device)
    
    # 逐个生成标记
    for i in range(max_len-1):
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

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400
    
    text = data['text']
    logger.info(f"Received text: {text}")
    
    try:
        prediction = predict(text)
        logger.info(f"Prediction: {prediction}")
        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Model Deployment')
    parser.add_argument('--model_path', type=str, default='output/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='config/training_config.json',
                        help='Path to training config file')
    parser.add_argument('--model_config_path', type=str, default='config/model_config.json',
                        help='Path to model config file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    args = parser.parse_args()
    
    # 初始化模型
    init_model(args.model_path, args.config_path, args.model_config_path)
    
    # 启动服务器
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
