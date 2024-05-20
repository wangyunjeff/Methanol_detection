import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.utils.data

# class CustomDataset(Dataset):
#     def __init__(self, filepath, scaler=None):
#         data = pd.read_csv(filepath)
#         self.features = data.iloc[:, :-1]
#         self.labels = data.iloc[:, -1]
#
#         if scaler is None:  # 如果没有提供scaler，就创建一个
#             self.scaler = StandardScaler()
#             self.features = self.scaler.fit_transform(self.features)
#         else:  # 否则，使用提供的scaler进行变换
#             self.scaler = scaler
#             self.features = self.scaler.transform(self.features)
#
#         self.features = torch.tensor(self.features, dtype=torch.float32)
#         self.labels = torch.tensor(self.labels.values, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, index):
#         return self.features[index], self.labels[index]
class CustomDataset(Dataset):
    def __init__(self, filepath, scaler=None):
        # Load data
        data = pd.read_csv(filepath)
        self.features = data.iloc[:, :-1]
        self.labels = data.iloc[:, -1]

        # Scaling
        if scaler is None:  # 如果没有提供scaler，就创建一个
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:  # 否则，使用提供的scaler进行变换
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        # Convert to PyTorch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Add a new dimension to match the shape [1, 3800]
        feature = self.features[index].unsqueeze(0)
        return feature, self.labels[index]
# def load_and_preprocess_data(filepath, scaler=None):
#     data = pd.read_csv(filepath)
#     features = data.iloc[:, :-1]
#     labels = data.iloc[:, -1]
#
#     if scaler is None:  # 如果没有提供scaler，就创建一个
#         scaler = StandardScaler()
#         features = scaler.fit_transform(features)
#     else:  # 否则，使用提供的scaler进行变换
#         features = scaler.transform(features)
#
#     features = torch.tensor(features, dtype=torch.float32)
#     labels = torch.tensor(labels.values, dtype=torch.long)
#
#     return TensorDataset(features, labels)


def load_alternative_data(data_path, scaler=None):
    # Example of a different data loading method
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

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


def load_triplet_data(filepath, focus_labels=None, focus_factors=None):
    if focus_factors is None:
        focus_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    if focus_labels is None:
        focus_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data = pd.read_csv(filepath)
    features = data.iloc[:, :-1].values  # Convert DataFrame to NumPy array
    labels = data.iloc[:, -1].values
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return TripletDataset(features, labels, focus_labels=focus_labels, focus_factors=focus_factors)


def load_and_preprocess_data(filepath, scaler=None):
    dataset = CustomDataset(filepath, scaler)
    return dataset

def get_data_loader(data_loader_name):
    if data_loader_name == "default":
        return load_and_preprocess_data
    elif data_loader_name == "Triplet":
        return load_triplet_data
    elif data_loader_name == "Triplet_with_exposed":
        return None
    else:
        raise ValueError(f"Unknown data loader name: {data_loader_name}")


if __name__ == '__main__':
    get_data_loader('default')
    pass
