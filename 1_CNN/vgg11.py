'''
VGG-11 neural network. Instantiate model with vgg.Net()
'''

import torch.nn as nn

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            vgg_block(1, 64),
            vgg_block(1, 128),
            vgg_block(2, 256),
            # nn.LazyBatchNorm2d(), # This shouldn't have been here, but it helps with gradient vanishing
            vgg_block(2, 512),
            vgg_block(2, 512),
            # CIFAR-10 has images of size 32x32. After 5 layers of MaxPool2d, they shrink to 1x1. Will this cause problem?
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(1024), nn.ReLU(), # Originally 4096. Changed to 1024 since CIFAR-10 has much less classes
            nn.LazyLinear(num_classes) # No need to add final softmax. CrossEntropyLoss does it for us.
        )
        
    def forward(self, x):
        return self.net(x)