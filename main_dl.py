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

# from config.config import config
# from config.config_triplet import config
from config.config_cnns import config
# from config.config_transformer import config

from data.dataloder import get_data_loader
from models.all_models import SimpleMLP
from models.losses import get_loss_function
from models.all_models import get_model
from utils.training_utils import evaluate_model, train_model, train_triplet_model

writer = SummaryWriter(config.general.log_dir)
# Check for available CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select the data loader function
data_loader_func = get_data_loader(config.general.data_loader_name)
# Load and preprocess data

train_dataset = data_loader_func(config.general.train_data_path)
test_dataset = data_loader_func(config.general.test_data_path)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.general.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.general.batch_size, shuffle=False)

input_size = train_dataset.features.shape[1]
hidden_size = config.model.hidden_size
num_classes = len(torch.unique(train_dataset.labels))

print(f"Input size: {input_size}")
print(f"Number of classes: {num_classes}")

model = get_model(config.model.model_name, input_size, hidden_size, num_classes, config.model.dropout_rate)

# model = SimpleMLP(input_size, config.model.hidden_size, num_classes, config.model.dropout_rate).to(device)

# Use weighted cross-entropy loss
criterion = get_loss_function(config.general.loss_name)

if config.general.optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

for epoch in range(config.general.num_epochs):
    if config.training.train_method == 'Triplet':
        train_triplet_model(model, train_loader, test_loader, criterion, optimizer, epoch, writer, num_epochs=10000)
    else:
        train_model(model, train_loader, criterion, optimizer, writer, epoch)  # Train for one epoch

        if (
                epoch + 1) % 10 == 0 or epoch == config.general.num_epochs - 1:  # Evaluate every 10 epochs and at the last epoch
            evaluate_model(model, test_loader, epoch, writer)

# 皮尔逊系数
# spernen correlation
