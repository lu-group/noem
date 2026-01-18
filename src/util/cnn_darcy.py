import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, activation1=F.relu, activation2=F.tanh, outputsize=128):
        super(CNN, self).__init__()
        # Assuming activation function passed as an argument (default is ReLU)
        self.activation1 = activation1
        self.activation2 = activation2

        # Define the layers
        self.reshape = nn.Unflatten(1, (1, 20, 20))  # Reshape input to (1, 20, 20)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=2)  # First convolution layer
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=2)  # Second convolution layer
        self.flatten = nn.Flatten()  # Flatten the output
        self.fc1 = nn.Linear(256, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, outputsize)
        self.fc3 = nn.Linear(outputsize, outputsize)  # Second fully connected layer

    def forward(self, x):
        x = self.reshape(x)  # Reshape input
        x = self.activation1(self.conv1(x))  # First conv layer with activation
        x = self.activation1(self.conv2(x))  # Second conv layer with activation
        x = self.flatten(x)  # Flatten the output
        x = self.activation2(self.fc1(x))  # First fully connected layer with activation
        x = self.activation2(self.fc2(x))
        # x = self.activation2(self.fc3(x))  # Second fully connected layer with activation
        return x
