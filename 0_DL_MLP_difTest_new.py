import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
# train_model(model, train_loader, criterion, optimizer)
from sklearn.metrics import confusion_matrix, classification_report

from config.config import config
from data.dataloder import load_and_preprocess_data
from models.all_models import SimpleMLP
from utils.training_utils import evaluate_model, train_model

writer = SummaryWriter('runs_SIN_difTest/MLP-hidden20-LR0.001')
# Check for available CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
scaler = None  # Start without a scaler
train_dataset, scaler = load_and_preprocess_data(config.general.train_data_path, scaler)
test_dataset, _ = load_and_preprocess_data(config.general.test_data_path, scaler)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.general.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.general.batch_size, shuffle=False)

input_size = train_dataset.tensors[0].shape[1]
num_classes = len(torch.unique(train_dataset.tensors[1]))

print(f"Input size: {input_size}")
print(f"Number of classes: {num_classes}")

# Create model instance, adjustable dropout_rate
model = SimpleMLP(input_size, config.model.hidden_size, num_classes, config.model.dropout_rate).to(device)

# Use weighted cross-entropy loss
criterion = nn.CrossEntropyLoss(weight=None)
optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

for epoch in range(config.general.num_epochs):
    train_model(model, train_loader, criterion, optimizer, config.writer, epoch)  # Train for one epoch
    if (epoch + 1) % 10 == 0 or epoch == config.general.num_epochs - 1:  # Evaluate every 10 epochs and at the last epoch
        evaluate_model(model, test_loader, epoch, config.writer)


# 皮尔逊系数
# spernen corrlation