import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, in_channels, h, w, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*h*w, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forwardpass(self,x):
        x = self.conv(x)
        x = self.lin(x)
        return x        