# DbAI-py : PyTorch Transformer 模型

这是一个使用PyTorch实现的Transformer模型项目，支持序列到序列的转换任务，如机器翻译、文本摘要等。

## 项目结构

```
transformer_project/
├── config/                # 配置文件
├── data/                  # 数据文件
├── docs/                  # 项目文档
├── models/                # 模型定义
├── utils/                 # 工具函数
├── main.py                # 项目主管理脚本
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── deploy.py              # 部署脚本
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
```

## 文档

- [训练指南](docs/training.md)
- [使用和部署指南](docs/usage.md)
- [实现说明](docs/how-to-achieve.md)

## 快速开始

1. 安装依赖(清华源镜像)：

```bash
pip3 install pytorch torchvision torchaudio flask tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 准备数据并配置参数

3. 使用 main.py 管理脚本训练模型：

```bash
python main.py train
```

4. 使用 main.py 管理脚本进行推理：

```bash
python main.py inference --model_path output/best_model.pth --input_file data/raw/test_data.json
```

5. 使用 main.py 管理脚本部署模型为API服务：

```bash
python main.py deploy --model_path output/best_model.pth
```

## 技术特点

- 完整实现Transformer架构
- 模块化设计，易于扩展
- 支持CPU和GPU计算
- 提供详细的训练、推理和部署文档
- 配置文件与代码分离，便于调整参数
- 新增主管理脚本，提供统一的命令行界面

## 贡献

欢迎贡献代码、文档或报告问题！请先阅读贡献指南。

这个项目提供了一个完整的Transformer模型实现，包含了训练、推理和部署的所有必要组件。您可以根据自己的需求调整配置文件中的参数，或扩展代码以支持更多功能。

这些更新后的文档现在提供了两种使用方式：使用 `main.py` 管理脚本或直接调用各个模块的脚本，使项目的使用更加灵活和便捷。