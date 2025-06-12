# DbAI-py: PyTorch Transformer Model

This is a project that implements a Transformer model using PyTorch, supporting sequence-to-sequence conversion tasks such as machine translation and text summarization.

## Languages

- [Chinese](docs/README-zh_cn.md)

## Project Structure

```
transformer_project/
├── config/                # Configuration files
├── data/                  # Data files
├── docs/                  # Project documentation
├── models/                # Model definitions
├── utils/                 # Utility functions
├── main.py                # Main management script for the project
├── train.py               # Training script
├── inference.py           # Inference script
├── deploy.py              # Deployment script
├── requirements.txt       # Dependency packages
└── README.md              # Project description
```

## Documentation

- [Training Guide](docs/training.md)
- [Usage and Deployment Guide](docs/usage.md)
- [Implementation Notes](docs/how-to-achieve.md)

## Quick Start

1. Install dependencies:

   ```bash
   pip3 install pytorch torchvision torchaudio flask tqdm
   ```

   Note: Due to network issues, the URL for the Tsinghua University mirror may not be accessible. If you encounter any problems, please check the validity of the URL and try again. Alternatively, you can use other PyPI mirrors or the default PyPI repository.

2. Prepare the data and configure the parameters.

3. Train the model using the `main.py` management script:

   ```bash
   python main.py train
   ```

4. Perform inference using the `main.py` management script:

   ```bash
   python main.py inference --model_path output/best_model.pth --input_file data/raw/test_data.json
   ```

5. Deploy the model as an API service using the `main.py` management script:

   ```bash
   python main.py deploy --model_path output/best_model.pth
   ```

## Technical Features

- Fully implements the Transformer architecture.
- Modular design for easy extension.
- Supports both CPU and GPU computation.
- Provides detailed documentation for training, inference, and deployment.
- Separates configuration files from code for convenient parameter adjustment.
- Introduces a main management script (`main.py`) to provide a unified command-line interface.

## Contribution

Contributions in the form of code, documentation, or bug reports are welcome! Please read the contribution guidelines first.

This project offers a complete implementation of the Transformer model, including all necessary components for training, inference, and deployment. You can adjust the parameters in the configuration files according to your needs or extend the code to support more features.

The updated documentation now provides two ways to use the project: through the `main.py` management script or by directly invoking the scripts of individual modules. This makes the project more flexible and convenient to use.
