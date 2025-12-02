import torch.nn as nn
import torch

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.35)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Adaptive pool to fixed size
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected
        x = self.dropout(torch.relu(self.fc1(x)))

        # Output classes
        x = self.fc2(x)

        return x
