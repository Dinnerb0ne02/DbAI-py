import torch
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def create_look_ahead_mask(size):
    mask = 1 - torch.triu(torch.ones(size, size))
    return mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
