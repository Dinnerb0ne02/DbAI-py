# 模型使用和部署指南

## 使用 main.py 管理脚本

项目现在提供了一个集中式的管理脚本 `main.py`，可以通过命令行参数选择不同的功能：

### 训练模型

```bash
python main.py train --batch_size 32 --num_epochs 100
```

### 模型推理

```bash
python main.py inference --model_path output/best_model.pth --input_file data/raw/test_data.json
```

### 部署模型

```bash
python main.py deploy --model_path output/best_model.pth --port 5001
```

### 配置管理

查看配置：
```bash
python main.py config --type training --show
```

交互式修改配置：
```bash
python main.py config --type model
```

### 运行自定义脚本

```bash
python main.py script --name my_custom_script.py
```

### 查看帮助信息

查看所有可用命令：
```bash
python main.py --help
```

查看特定命令的帮助：
```bash
python main.py train --help
```

## 传统使用方法

如果您更习惯直接调用各个模块的脚本，仍然可以按照以下方式使用：

### 模型推理

要使用训练好的模型进行推理，请运行以下命令：

```bash
python inference.py --model_path output/best_model.pth --input_file data/raw/test_data.json --output_file output/predictions.json
```

您可以通过以下参数自定义推理过程：

- `--config`: 训练配置文件路径
- `--model_config`: 模型配置文件路径
- `--model_path`: 模型文件路径
- `--input_file`: 输入数据文件路径
- `--output_file`: 输出预测结果文件路径

### 模型部署

要在本地部署模型作为API服务，请运行以下命令：

```bash
python deploy.py --model_path output/best_model.pth --host 0.0.0.0 --port 5000
```

您可以通过以下参数自定义部署过程：

- `--model_path`: 模型文件路径
- `--config_path`: 训练配置文件路径
- `--model_config_path`: 模型配置文件路径
- `--host`: 服务器主机地址
- `--port`: 服务器端口

### API使用示例

部署成功后，您可以通过发送POST请求到`/predict`端点来使用模型：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "your input text here"}' http://localhost:5000/predict
```

您也可以使用Python代码调用API：

```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "your input text here"}
response = requests.post(url, json=data)

if response.status_code == 200:
    print(response.json()["prediction"])
else:
    print(f"Error: {response.status_code}")
```

### 配置文件说明

模型的配置文件位于`config`目录下，包括：

- `model_config.json`: 模型架构参数
- `training_config.json`: 训练参数

您可以根据需要修改这些配置文件来自定义模型和训练过程。


