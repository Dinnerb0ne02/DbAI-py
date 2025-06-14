
# 模型训练指南

## 准备数据
在开始训练前，您需要准备好数据集。数据集应该包含源语言和目标语言的文本对，并按照以下格式组织：

```plaintext
data/
├── processed/
│   ├── src_vocab.pkl
│   ├── tgt_vocab.pkl
│   ├── train_data.json
│   └── val_data.json
└── raw/
    └── your_raw_data_files
```
训练和验证数据文件应该是JSON格式，包含两个键：`src` 和 `tgt`，分别对应源语言和目标语言的文本列表。

## 配置训练参数
您可以通过修改 `config/training_config.json` 和 `config/model_config.json` 来配置训练参数和模型参数。


## 开始训练
要开始训练模型，请运行以下命令：
```bash
python main.py train
```
或者直接运行训练脚本：
```bash
python train.py
```
训练过程中，模型会定期保存到 `output` 目录中。您可以通过TensorBoard查看训练进度：
```bash
tensorboard --logdir=output/tensorboard
```

## 训练过程说明
训练过程包括以下步骤：
1. **加载和预处理数据**：从指定的数据目录中加载训练和验证数据，并进行必要的预处理，如将文本转换为索引序列、添加开始和结束标记、截断或填充序列等。
2. **初始化模型、优化器和损失函数**：根据配置文件中的参数初始化Transformer模型，选择合适的优化器（如Adam）和损失函数（如交叉熵损失）。
3. **执行训练循环**：在每个训练轮次中，模型进行前向传播、损失计算、反向传播和参数更新。同时，使用学习率调度器和梯度裁剪来提高训练稳定性。
4. **定期评估模型性能并保存最佳模型**：每隔一定轮次，在验证集上评估模型的性能（如损失和BLEU分数），并保存性能最佳的模型。


## 训练参数说明

- `data_path`: 数据目录路径
- `output_dir`: 输出目录路径
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `clip`: 梯度裁剪阈值
- `eval_every`: 每隔多少轮评估一次模型
- `save_every`: 每隔多少轮保存一次模型

## 模型参数说明

- `d_model`: 模型维度
- `num_heads`: 多头注意力的头数
- `num_encoder_layers`: 编码器层数
- `num_decoder_layers`: 解码器层数
- `d_ff`: 前馈网络的维度
- `max_seq_length`: 最大序列长度
- `dropout`: Dropout率