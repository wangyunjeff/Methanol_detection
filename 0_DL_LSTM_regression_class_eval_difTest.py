import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter('runs_DL_LSTM_regression_IN_difTest/LSTM-hidden20-LR0.001')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading and preprocessing
def load_and_preprocess_data(filepath, scaler=None):
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    label_mapping = {0: 0, 1: 2.4, 2: 6, 3: 12, 4: 24, 5: 48, 6: 72, 7: 96, 8: 120}
    labels = labels.map(label_mapping)

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Adding sequence dimension
    labels = torch.tensor(labels.values, dtype=torch.float32)

    return TensorDataset(features, labels), scaler


scaler = None
train_dataset, scaler = load_and_preprocess_data('IN_train_data_with_labels_difTest.csv', scaler)
test_dataset, _ = load_and_preprocess_data('IN_test_data_with_labels_difTest.csv', scaler)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# LSTM Model definition
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # We only use the last LSTM output
        return self.fc(x)


model = SimpleLSTM(1000, 20).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def regression_to_classification(y_pred, thresholds):
    class_labels = np.digitize(y_pred, thresholds, right=True)
    return (class_labels).astype(int)  # Adjust because np.digitize starts from 1 and ensure integer type

def inverse_mapping(value):
    if value < 1.2:
        return 0
    elif value < 4.2:
        return 1
    elif value < 9:
        return 2
    elif value < 18:
        return 3
    elif value < 36:
        return 4
    elif value < 60:
        return 5
    elif value < 84:
        return 6
    elif value < 108:
        return 7
    else:
        return 8


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, thresholds=None):
    ave_losses = []
    for epoch in range(num_epochs):
        model.train()
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
            evaluate_model(model, test_loader, epoch, thresholds)
            output_predictions(model, test_loader, epoch)
            model.train()


# Evaluation function
def evaluate_model(model, test_loader, epoch, thresholds):
    model.eval()
    total_loss = 0
    all_actuals = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

            # Convert regression predictions to class labels
            # Convert regression predictions to class labels
            predicted_labels = regression_to_classification(outputs.squeeze().cpu().numpy(), thresholds)
            all_predictions.extend(predicted_labels)

            # Apply inverse mapping to actual labels
            actual_class_labels = [inverse_mapping(label.item()) for label in labels.cpu().numpy()]
            all_actuals.extend(actual_class_labels)

    # Calculate classification metrics
    acc = accuracy_score(all_actuals, all_predictions)
    cm = confusion_matrix(all_actuals, all_predictions)
    cr = classification_report(all_actuals, all_predictions)

    average_loss = total_loss / len(test_loader)
    writer.add_scalar('Loss/test', average_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)
    print(f"Test Loss: {average_loss}")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)
    return average_loss, acc, cm, cr


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

    # Computing averages and standard deviations
    categories = sorted(predictions_dict.keys())
    avg_predictions = [np.mean(predictions_dict[label]) for label in categories]
    std_predictions = [np.std(predictions_dict[label]) for label in categories]

    # Plotting
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
    plt.savefig(f'runs_DL_LSTM_regression_IN_difTest/{epoch}.png')
# Define thresholds based on the midpoint between labeled class values
thresholds = [1.2, 4.2, 9, 18, 36, 60, 84, 108]
# thresholds = {0, 2.4, 6, 12, 24, 48, 72, 96, 120}

# Execute training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=10000, thresholds=thresholds)
