import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(filepath, exclude_label=None, scaler=None):
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    label_mapping = {0: 0, 1: 2.4, 2: 6, 3: 12, 4: 24, 5: 48, 6: 72, 7: 96, 8: 120}
    labels = labels.map(label_mapping)

    if exclude_label is not None:
        exclude_indices = labels.isin(exclude_label)
        features = features[~exclude_indices]
        labels = labels[~exclude_indices]

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.float32)

    return TensorDataset(features, labels), scaler

# Initialize SummaryWriter
writer = SummaryWriter('runs_regression_SIN_difTest/MLP-hidden20-LR0.001')

# Load and preprocess data
scaler = None
train_dataset, scaler = load_and_preprocess_data('SIN_train_data_with_labels(regression)_difTest.csv', exclude_label=[24], scaler=scaler)
test_dataset, _ = load_and_preprocess_data('SIN_test_data_with_labels(regression)_difTest.csv', scaler=scaler)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

model = SimpleMLP(train_dataset.tensors[0].shape[1], 20, 0.0).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
            evaluate_model(model, test_loader, epoch)
            output_predictions(model, test_loader, epoch)

# Evaluation function
def evaluate_model(model, test_loader, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    writer.add_scalar('Loss/test', average_loss, epoch)
    print(f"Test Loss: {average_loss}")

# Prediction and plotting function
def output_predictions(model, test_loader, epoch):
    model.eval()
    predictions_dict = {}
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = outputs.squeeze()

            actuals = labels.cpu().numpy()
            predictions = predicted.cpu().numpy()

            for actual, prediction in zip(actuals, predictions):
                if actual in predictions_dict:
                    predictions_dict[actual].append(prediction)
                else:
                    predictions_dict[actual] = [prediction]

    categories = sorted(predictions_dict.keys())
    avg_predictions = [np.mean(predictions_dict[label]) for label in categories]
    std_predictions = [np.std(predictions_dict[label]) for label in categories]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(categories))

    plt.bar(index, categories, bar_width, label='Actual Value')
    plt.bar(index + bar_width, avg_predictions, bar_width, yerr=std_predictions, label='Average Prediction', alpha=0.6, capsize=5)

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Comparison of Actual and Average Predicted Values per Category')
    plt.xticks(index + bar_width / 2, [f'Label {label}' for label in categories])
    plt.legend()
    plt.grid(True)
    plt.savefig(f'runs_regression_SIN_difTest/{epoch}.png')
    plt.close()

# Execute training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=10000)
