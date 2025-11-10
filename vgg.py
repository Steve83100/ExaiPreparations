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
        print("Created model: vgg")
        self.net = nn.Sequential(
            vgg_block(1, 64),
            vgg_block(1, 128),
            vgg_block(2, 256),
            vgg_block(2, 512),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(num_classes) # No need to add final softmax. CrossEntropyLoss does it for us.
        )
        
    def forward(self, x):
        return self.net(x)