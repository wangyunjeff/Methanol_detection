import math

import torch
import torch.nn as nn
from config.config import config
import torch.nn.functional as F

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
        x = x.view(x.size(0), -1)  # This reshapes the input to remove the channel dimension

        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=300, stride=100, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=300, stride=100, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=300, stride=100, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Use a dummy input to calculate the size for the first fully connected layer
        dummy_input = torch.zeros(1, 1, 3800)
        output_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(output_size, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        return x.numel()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_model=512, num_heads=8, num_encoder_layers=6, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Embedding layer for time series data
        self.embedding = nn.Linear(num_features, dim_model)

        # Create positional encoding
        self.positional_encoding = self.create_positional_encoding(max_len=5000, dim_model=dim_model)

        # Transformer Encoder Configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True  # This ensures the input is expected in [batch, seq, feature] format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer for classification
        self.output = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        # Add embedding to input
        x = self.embedding(x)  # Shape: [batch_size, seq_len, dim_model]

        # Apply positional encoding, adjusted for batch size
        pos_encoding = self.positional_encoding[:x.size(1), :].unsqueeze(0)
        pos_encoding = pos_encoding.expand(x.size(0), -1, -1)  # Ensure pos_encoding matches batch size
        x = x + pos_encoding  # Broadcasting happens here

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate output, for example, taking the mean across the sequence dimension
        x = x.mean(dim=1)

        # Classifier layer to predict the class
        x = self.output(x)
        return x

    def create_positional_encoding(self, max_len, dim_model):
        """Generate positional encoding."""
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

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

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        # Key, Query, and Value transformations
        self.key = nn.Linear(feature_dim, feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, x):
        # Shape of x: [batch_size, seq_len, feature_dim]
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        attended = torch.bmm(attention_weights, values)
        return attended

class AttentionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(AttentionMLP, self).__init__()
        self.attention = SelfAttention(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()  # using Sigmoid as per your previous setup
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Assuming x is [batch_size, seq_len, feature_dim]
        # Apply self-attention
        x = self.attention(x)  # [batch_size, seq_len, feature_dim]

        # Aggregate across the sequence dimension (e.g., by summing or averaging)
        x = x.mean(dim=1)  # [batch_size, feature_dim]

        # MLP processing
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_model(model_name, input_size, hidden_size, num_classes, dropout_rate):
    if model_name == "mlp":
        return SimpleMLP(input_size, hidden_size, num_classes, dropout_rate)
    # Add other model instantiations here
    elif model_name == "SimpleMLP_triplet":
        return SimpleMLP_triplet(input_size, hidden_size, dropout_rate)
    elif model_name == "SimpleCNN":
        return SimpleCNN(num_classes)
    elif model_name == "TimeSeriesTransformer":
        return TimeSeriesTransformer(input_size, num_classes,)
    elif model_name == "AttentionMLP":
        return AttentionMLP(input_size, hidden_size, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    model = get_model(config.model.model_name, 100, config.model.hidden_size, 10, config.model.dropout_rate)
    print(model)
    pass
