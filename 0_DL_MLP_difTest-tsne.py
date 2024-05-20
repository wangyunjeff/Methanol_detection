import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs_SIN_difTest-tsne/MLP-hidden20-LR0.001')
# 检查是否有可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义函数加载和预处理数据
def load_and_preprocess_data(filepath, scaler=None):
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    if scaler is None:  # 如果没有提供scaler，就创建一个
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:  # 否则，使用提供的scaler进行变换
        features = scaler.transform(features)

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.long)

    return TensorDataset(features, labels), scaler


scaler = None  # 开始时没有scaler
train_dataset, scaler = load_and_preprocess_data('./SIN_train_data_with_labels_difTest.csv', scaler)
# validation_dataset, _ = load_and_preprocess_data('./validation_data.csv', scaler)
test_dataset, _ = load_and_preprocess_data('./SIN_test_data_with_labels_difTest.csv', scaler)

# 创建DataLoaders
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch.nn as nn

# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         # self.relu = nn.ReLU()
#         self.relu = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc3(out)
#         return out

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embedding=True):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        embedding = self.relu(out)  # This will be our embedding
        out = self.dropout(embedding)
        out = self.fc3(out)
        if return_embedding:
            return out, embedding
        return out





# ????????
# input_size = 1200  # ????????
# num_classes = len(labels.unique())  # ???????
# num_heads = 4  # ????????
# dim_feedforward = 512  # ??????
# num_layers = 2  # Transformer ??????
#
# model = SimplifiedTransformer(input_size, num_classes, num_heads, dim_feedforward, num_layers).to(device)
# 实例化模型、损失函数和优化器
# model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=len(labels.unique())).to(device)
input_size = train_dataset.tensors[0].shape[1]
hidden_size = 20  # 这是一个可调整的参数
num_classes = len(torch.unique(train_dataset.tensors[1]))

print(f"Input size: {input_size}")
print(f"Number of classes: {num_classes}")
# 创建模型实例，可以调整 dropout_rate 的值
model = SimpleMLP(input_size, hidden_size, num_classes, dropout_rate=0.0).to(device)
# model = SimpleMLP(input_size, hidden_size, num_classes, dropout_rate=0.0).to(device)
# 计算类别权重
# class_counts = data['Label'].value_counts().sort_index().values
# class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
# class_weights /= class_weights.sum()
# class_weights = class_weights.to(device)

# 使用加权交叉熵损失
criterion = nn.CrossEntropyLoss(weight=None)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # 输入不需要增加维度
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        if epoch%100==0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
            # evaluate_model(model, test_loader, epoch)
            evaluate_model_with_embeddings(model, test_loader, writer, epoch)

# train_model(model, train_loader, criterion, optimizer)
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, epoch):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    for class_label in report:
        if class_label.isdigit():
            writer.add_scalar(f'Accuracy/class_{class_label}', report[class_label]['recall'], epoch)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

from torch.utils.tensorboard import SummaryWriter
import numpy as np

def evaluate_model_with_embeddings(model, test_loader, writer, epoch):
    model.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, embedding = model(inputs, return_embedding=True)
            embeddings.append(embedding.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, 0)
    labels_list = np.concatenate(labels_list, 0)

    # 添加到TensorBoard
    writer.add_embedding(embeddings, metadata=labels_list, global_step=epoch, tag="Embeddings")

# 确保TensorBoard writer已初始化
# writer = SummaryWriter('runs/your_experiment_name')

# 在适当的时刻调用evaluate_model_with_embeddings
# 例如，在每个epoch结束时或训练结束后


# 调用训练和评估函数
train_model(model, train_loader, criterion, optimizer, num_epochs=10000)
