
# Define the CNN architecture
import torch
import torch.optim as optim
import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,20)
        self.fc4 = nn.Linear(20,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
    
class CNNWithSkipConnections(nn.Module):
    def __init__(self):
        super(CNNWithSkipConnections, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding='same', activation=nn.ReLU())
        self.pool1 = nn.MaxPool2d((2, 2), strides=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same', activation=nn.ReLU())
        self.pool2 = nn.MaxPool2d((2, 2), strides=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding='same', activation=nn.ReLU())
        self.pool3 = nn.MaxPool2d((2, 2), strides=(2, 2))

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 7)
        self.softmax = nn.Softmax(dim=1)

        # Skip connections
        self.skip1 = nn.Conv2d(32, 128, (1, 1), padding='same', activation=nn.ReLU())
        self.skip2 = nn.Conv2d(64, 128, (1, 1), padding='same', activation=nn.ReLU())

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.skip1(x) + x

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.skip2(x) + x

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.softmax(x)
        return x

