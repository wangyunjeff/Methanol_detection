import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Example usage
input_size = 3800  # feature dimension
hidden_size = 512  # hidden dimension for MLP
num_classes = 10  # example number of classes
dropout_rate = 0.5
model = AttentionMLP(input_size, hidden_size, num_classes, dropout_rate)

# Example input
example_input = torch.randn(64, 1, 3800)  # [batch_size, seq_len, feature_dim]
output = model(example_input)
print(output.shape)  # Expected output shape: [64, num_classes]
