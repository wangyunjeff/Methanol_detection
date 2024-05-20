import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

# 检查是否有可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer1 = SummaryWriter('runs_IN/MLP100-triplet1-withexposure')
# writer1 = SummaryWriter('runs/features1_0.5')
# writer2 = SummaryWriter('runs/model2')


from torch.utils.data import Dataset
import numpy as np

# class TripletDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
#         self.labels_set = list(set(labels.numpy()))
#         self.label_to_indices = {label: np.where(labels.numpy() == label)[0]
#                                  for label in self.labels_set}
#
#     def __getitem__(self, index):
#         anchor_feature = self.features[index]
#         anchor_label = self.labels[index].item()  # Convert tensor to Python int
#         positive_index = index
#         while positive_index == index:
#             positive_index = np.random.choice(self.label_to_indices[anchor_label])
#         negative_label = np.random.choice(list(set(self.labels_set) - set([anchor_label])))
#         negative_index = np.random.choice(self.label_to_indices[negative_label])
#         positive_feature = self.features[positive_index]
#         negative_feature = self.features[negative_index]
#         return anchor_feature, positive_feature, negative_feature, anchor_label
#
#     def __len__(self):
#         return len(self.features)

# class TripletDataset(Dataset):
#     def __init__(self, features, labels, focus_label=1, focus_factor=0.5):
#         """
#         focus_label: The category to focus on.
#         focus_factor: Probability to pick a triplet involving the focus label as anchor or positive.
#         """
#         self.features = features
#         self.labels = labels
#         self.labels_set = list(set(labels.numpy()))
#         self.label_to_indices = {label: np.where(labels.numpy() == label)[0]
#                                  for label in self.labels_set}
#         self.focus_label = focus_label
#         self.focus_factor = focus_factor
#
#     def __getitem__(self, index):
#         anchor_feature = self.features[index]
#         anchor_label = self.labels[index].item()
#         if np.random.rand() < self.focus_factor and self.focus_label in self.labels_set:
#             # With focus_factor probability, force anchor or positive to be focus_label
#             if anchor_label != self.focus_label:
#                 index = np.random.choice(self.label_to_indices[self.focus_label])
#                 anchor_feature = self.features[index]
#                 anchor_label = self.focus_label
#
#         positive_index = index
#         while positive_index == index:
#             positive_index = np.random.choice(self.label_to_indices[anchor_label])
#         negative_label = np.random.choice(list(set(self.labels_set) - set([anchor_label])))
#         negative_index = np.random.choice(self.label_to_indices[negative_label])
#         positive_feature = self.features[positive_index]
#         negative_feature = self.features[negative_index]
#         return anchor_feature, positive_feature, negative_feature, anchor_label
#
#     def __len__(self):
#         return len(self.features)

class TripletDataset(Dataset):
    def __init__(self, features, labels, focus_labels=None, focus_factors=None):
        """
        focus_labels: A list of categories to focus on. Each element must be unique.
        focus_factors: A list of probabilities corresponding to each focus label, determining the
                       likelihood of picking a triplet involving the focus label as anchor or positive.
        """
        self.features = features
        self.labels = labels
        self.labels_set = list(set(labels.numpy()))
        self.label_to_indices = {label: np.where(labels.numpy() == label)[0]
                                 for label in self.labels_set}

        if focus_labels is None:
            focus_labels = []
        if focus_factors is None:
            focus_factors = []

        assert len(focus_labels) == len(focus_factors), "focus_labels and focus_factors must have the same length"

        # Normalize focus_factors if they do not sum to 1
        total_focus = sum(focus_factors)
        if total_focus > 0:
            focus_factors = [f / total_focus for f in focus_factors]

        self.focus_labels = focus_labels
        self.focus_factors = focus_factors
        self.focus_map = dict(zip(focus_labels, focus_factors))

    def __getitem__(self, index):
        anchor_feature = self.features[index]
        anchor_label = self.labels[index].item()

        # Decide to enforce focus based on focus labels and their factors
        if self.focus_labels:
            # Draw a random number to decide if we need to switch focus
            rnd_focus_draw = np.random.rand()
            cumulative_focus = 0
            for label, factor in zip(self.focus_labels, self.focus_factors):
                cumulative_focus += factor
                if rnd_focus_draw < cumulative_focus:
                    # Switch focus to this label
                    focus_label = label
                    if anchor_label != focus_label:
                        index = np.random.choice(self.label_to_indices[focus_label])
                        anchor_feature = self.features[index]
                        anchor_label = focus_label
                    break

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        negative_label = np.random.choice(list(set(self.labels_set) - set([anchor_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        positive_feature = self.features[positive_index]
        negative_feature = self.features[negative_index]
        return anchor_feature, positive_feature, negative_feature, anchor_label

    def __len__(self):
        return len(self.features)


import torch.nn as nn


class SimpleMLP_triplet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(SimpleMLP_triplet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Removed the final classification layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        # Return embeddings before dropout
        return out


# 数据加载和预处理
data = pd.read_csv('./merged_INdata_with_labels.csv')
features = data.iloc[:, :-1].values  # Convert DataFrame to NumPy array
labels = data.iloc[:, -1].values
# Convert features and labels to PyTorch tensors if not already done
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

focus_labels = [0, 1, 2,3,4,5,6,7,8]  # The categories you want to focus on
focus_factors = [0.3, 0.2, 0.2,0.1,0.1,0.1,0.1,0.1,0.1]  # Corresponding focus factors for each category
# focus_factors = [1,1,1,1,1,1,1,1,1]  # Corresponding focus factors for each category
# focus_labels = [1]  # The categories you want to focus on
# focus_factors = [0.5]  # Corresponding focus factors for each category

# Assuming features and labels are already defined
triplet_dataset = TripletDataset(features, labels, focus_labels=focus_labels, focus_factors=focus_factors)
# triplet_dataset = TripletDataset(features, labels)
# Create dataset and data loader
# triplet_dataset = TripletDataset(features, labels)
triplet_loader = DataLoader(triplet_dataset, batch_size=64, shuffle=True)

from torch.utils.data import random_split

# Assuming `triplet_dataset` is an instance of TripletDataset containing all your data
dataset_size = len(triplet_dataset)
train_size = int(0.7 * dataset_size)  # 80% for training
val_size = dataset_size - train_size  # 20% for validation

# Split the dataset
train_dataset, val_dataset = random_split(triplet_dataset, [train_size, val_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


input_size = features.size()[1]
hidden_size = 100  # 这是一个可调整的参数
num_classes = len(labels.unique())
print(len(labels.unique()))
# 创建模型实例，可以调整 dropout_rate 的值

model = SimpleMLP_triplet(input_size, hidden_size, dropout_rate=0.0).to(device) # Make sure it's suitable for triplet output
# 计算类别权重
# class_counts = data['Label'].value_counts().sort_index().values
# class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
# class_weights /= class_weights.sum()
# class_weights = class_weights.to(device)

# 使用加权交叉熵损失
# criterion = nn.CrossEntropyLoss(weight=None)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def train_triplet_model(model, train_loader, val_loader, triplet_loss, optimizer, num_epochs=10):
    # Initialize k-NN classifier (consider fitting it with training set embeddings outside this function)
    knn = KNeighborsClassifier(n_neighbors=3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative, _ in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        writer1.add_scalar('Loss/train', running_loss/ len(train_loader), epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")

        # Validation phase for loss
        model.eval()
        val_loss = 0.0
        embeddings = []
        val_labels = []
        with torch.no_grad():
            for anchor, positive, negative, labels in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embed = model(anchor)
                val_loss += triplet_loss(anchor_embed, model(positive), model(negative)).item()

                # Collect embeddings and labels for classification
                embeddings.extend(anchor_embed.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")

        # Fit k-NN on the collected embeddings and evaluate
        embeddings = np.array(embeddings)
        val_labels = np.array(val_labels)

        # Assuming you have pre-fitted the k-NN with training embeddings
        knn.fit(embeddings, val_labels)  # This should ideally be fitted with separate training embeddings
        y_pred = knn.predict(embeddings)
        report = classification_report(val_labels, y_pred, output_dict=True)
        for class_label in report:
            if class_label.isdigit():
                writer1.add_scalar(f'Accuracy/class_{class_label}', report[class_label]['recall'], epoch)

        print(confusion_matrix(val_labels, y_pred))
        print(classification_report(val_labels, y_pred))

train_triplet_model(model, train_loader, val_loader, triplet_loss, optimizer, num_epochs=10000)
