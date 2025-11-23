'''
Resnet-18 neural network. Instantiate model with resnet.Net()
'''

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, use_1x1conv = False, stride = 1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size = 3, padding = 1, stride = stride)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv: # Adjust residual's size, used when output's shape (c, h, w) differs from input
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size = 1, stride = stride)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
class ResidualModule(nn.Module): # Many (usually 2) residual blocks compose one module
    def __init__(self, num_blocks, num_channels, is_first_module = False):
        super().__init__()
        if(is_first_module):
            blocks = [ResidualBlock(num_channels) for _ in range(num_blocks)] # Pooling was done just now, so no need to half size
        else:
            blocks = [ResidualBlock(num_channels) for _ in range(num_blocks-1)]
            blocks.insert(0, ResidualBlock(num_channels, use_1x1conv=True, stride=2)) # Half size of image (both h and w)
        self.module = nn.Sequential(*blocks)
        
    def forward(self, X):
        return self.module(X)
        
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.begin = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1), # Kernel shrinked to adapt to the smaller cifar-10
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.mid = nn.Sequential(
            ResidualModule(2, 64, is_first_module=True),
            ResidualModule(2, 128),
            ResidualModule(2, 256),
            ResidualModule(2, 512)
        )
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
    def forward(self, x):
        return self.end(self.mid(self.begin(x)))