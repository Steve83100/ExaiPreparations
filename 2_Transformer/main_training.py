from __future__ import unicode_literals, print_function, division
from EngFra import *
from Transformer import *
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import random_split, DataLoader, RandomSampler

BATCH_SIZE = 64
TRAIN_RATIO = 0.8
torch.manual_seed(42)

if torch.cuda.is_available(): # Use CUDA if available
    DEVICE = torch.device("cuda:4")
    print("Using device: " + torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    print("Using device: CPU")



# ================================================================================
# Create training and validating dataloaders

def get_dataloaders():
    dataset = EnFrDataset()
    
    # Split dataset into train and validation
    train_size = int(TRAIN_RATIO * len(dataset))
    valid_size = len(dataset) - train_size # Remaining size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, shuffle = False, batch_size = BATCH_SIZE)
    
    return train_loader, valid_loader



# ================================================================================
# Instantiate and train model

train_loader, valid_loader = get_dataloaders()
input, input_len, target_S, target_E = next(iter(train_loader))
print(input.shape)
print(input.len)
print(target_S.shape)
print(target_E.shape)