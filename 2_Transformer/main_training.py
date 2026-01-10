from __future__ import unicode_literals, print_function, division
from EngFra import *
from Transformer import *

import math
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, RandomSampler
from torchinfo import summary



MAX_LEN = 10 # By default keeps sentences no longer than 10 words
BATCH_SIZE = 64
TRAIN_RATIO = 0.9
LR = 0.001
EPOCHS = 3
torch.manual_seed(42)

if torch.cuda.is_available(): # Use CUDA if available
    DEVICE = torch.device("cuda:3")
    print("Using device: " + torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    print("Using device: CPU")



# ================================================================================
# Create training and validating dataloaders

def get_dataloaders(max_len, train_ratio):
    dataset = EnFrDataset(max_len)
    
    # Split dataset into train and validation
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size # Remaining size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, shuffle = False, batch_size = BATCH_SIZE)
    
    return dataset, train_loader, valid_loader



# ================================================================================
# Define training and validating functions

def train(dataloader, model, device, criterion, optimizer):
    model.train()
    total_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        
        input, input_len, target_S, target_E = data
        input.to(device)
        input_len.to(device)
        target_S.to(device)
        target_E.to(device)
        
        output = model(input, target_S, input_len, MAX_LEN)
        # output shape: (b, n, v), but CEL expects (batch, num_classes, d1) which is (b, v, n)
        # target shape: (b, n), satisfies CEL expectation
        loss = criterion(torch.transpose(output, 1, 2), target_E)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(dataloader, model, device, criterion):
    model.eval()
    total_loss = 0
    for data in dataloader:
        input, input_len, target_S, target_E = data
        start_token = torch.ones((BATCH_SIZE, 1), dtype=int)
        input.to(device)
        input_len.to(device)
        start_token.to(device)
        target_E.to(device)
        
        output = model(input, start_token, input_len, MAX_LEN, True)
        loss = criterion(torch.transpose(output, 1, 2), target_E)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
    


# ================================================================================
# Create dataloader and model
    
dataset, train_loader, valid_loader = get_dataloaders(MAX_LEN, TRAIN_RATIO)

model = Transformer(
    input_vocab_size=dataset.input_lang.n_words,
    output_vocab_size=dataset.output_lang.n_words,
    embed_dim=32,
    ffn_num_hiddens=64,
    num_heads=8,
    num_layers=6,
    dropout=0.5
)

# print("Testing...")
# input, len_input, target_S, target_E = next(iter(train_loader)) # input and target are all word indexes, not one-hot vectors
# summary(model, input_data = [input, target_S, input_len])

# output = model(input, target_S, len_input, MAX_LEN) # teacher-forcing
# _, indexes = torch.max(output[0], axis=1) # Greedy sampling
# print([dataset.decode(index.item(), En=False) for index in indexes])

# start = torch.zeros((BATCH_SIZE, 1), dtype=int)
# output = model(input, start, len_input, MAX_LEN, True) # auto-regression
# _, indexes = torch.max(output[0], axis=1) # Greedy sampling
# print([dataset.decode(index.item(), En=False) for index in indexes])

model.to(DEVICE)
    


# ================================================================================
# Train and validate model

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

print("\n------------------------------------------------------")
print("Training begins")

start_time = time.time()
train_losses = []
valid_losses = []

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    train_loss = train(train_loader, model, DEVICE, criterion, optimizer)
    valid_loss = validate(train_loader, model, DEVICE, criterion)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    time_spent = asMinutes(time.time() - start_time)
    
    print(f'Epoch [{epoch}/{EPOCHS}]. Time spent:{time_spent}\n\tTrain Loss: {train_loss}. Valid Loss: {valid_loss}')
    
print("Training complete!")
print("------------------------------------------------------")



# ================================================================================
# Plot progress

plt.figure()
fig, ax = plt.subplots()
# this locator puts ticks at regular intervals
loc = ticker.MultipleLocator(base=0.2)
ax.yaxis.set_major_locator(loc)
plt.plot(train_losses)
plt.plot(valid_losses)
    


# ================================================================================
# Visualize attention