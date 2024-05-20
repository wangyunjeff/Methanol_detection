import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs_IN/MLP-hidden20-LR0.001')
# 检查是否有可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
data = pd.read_csv('./merged_IN_data_with_labels.csv')
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

scaler = StandardScaler()
features = scaler.fit_transform(features)

features = torch.tensor(features, dtype=torch.float32).to(device)

labels = torch.tensor(labels.values, dtype=torch.long).to(device)


dataset = TensorDataset(features, labels)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.relu = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
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
input_size = features.size()[1]
hidden_size = 20  # 这是一个可调整的参数
num_classes = len(labels.unique())
print(len(labels.unique()))
# 创建模型实例，可以调整 dropout_rate 的值
model = SimpleMLP(input_size, hidden_size, num_classes, dropout_rate=0.0).to(device)
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
            evaluate_model(model, test_loader, epoch)

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


# 调用训练和评估函数
train_model(model, train_loader, criterion, optimizer, num_epochs=10000)
