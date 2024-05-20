import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=30, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduces size by half

        # Convolutional layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=30, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Reduces size by half

        # Convolutional layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=30, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Reduces size by half

        # Flatten layer
        self.flatten = nn.Flatten()
        # Dummy input to calculate the input size of the first fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 3800)  # [batch_size, channels, length]
            output_size = self._get_conv_output(dummy_input)

        # Define the fully connected layers using the calculated output size
        self.fc1 = nn.Linear(output_size, 128)

        self.relu4 = nn.ReLU()

        # Fully connected layer 2 (Output layer)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
model = SimpleCNN(num_classes=10)  # Adjust num_classes as needed for your application
print(model)

# Test with a dummy input to check dimensions
dummy_input = torch.randn(64, 1, 3800)  # [batch_size, channels, sequence_length]
output = model(dummy_input)
print(output.shape)  # Expected output: [batch_size, num_classes]
