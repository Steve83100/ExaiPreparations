from __future__ import unicode_literals, print_function, division
from EngFra import *
from Transformer import *

import math
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, RandomSampler
from torchinfo import summary



MODEL_NAME = "Transformer"
DATASET_NAME = "EnFr"
ROOT_PATH = "/home/hqdeng7/syang/ExaiPreparations/" # Path of ExaiPreparations
MODEL_PATH = ROOT_PATH + "ModelSaves/"
OUTPUT_PATH = ROOT_PATH + "Outputs/"

MAX_LEN = 20 # By default keeps sentences no longer than 10 words
# Encoder-decoder structure allows us to delegate a different output length than input length
# But in this simple translation task we will keep output length and input length the same
# Which means that we will also use MAX_LEN as the "out_len" for decoder

BATCH_SIZE = 64
TRAIN_RATIO = 0.9
LR = 0.001
EPOCHS = 50
torch.manual_seed(86)

if torch.cuda.is_available(): # Use CUDA if available
    DEVICE = torch.device("cuda:3")
    print("Using device: " + torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    print("Using device: CPU")



# ================================================================================
# Define training and validating functions

def train(dataloader, model, device, criterion, optimizer):
    model.train()
    total_loss = 0
    for input, input_len, target_S, target_E in dataloader:
        optimizer.zero_grad()
        
        input = input.to(device)
        input_len = input_len.to(device)
        target_S = target_S.to(device)
        target_E = target_E.to(device)
        
        output = model(input, target_S, input_len, MAX_LEN) # Just use MAX_LEN as out_len
        # output shape: (b, n, v), but CEL expects (batch, num_classes, d1) which is (b, v, n), so we transpose
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
    for input, input_len, target_S, target_E in dataloader:
        # start_token = torch.ones((BATCH_SIZE, 1), dtype=int)
        # You should NOT use BATCH_SIZE, as the last batch usually has less than BATCH_SIZE samples!
        start_token = torch.ones((input.shape[0], 1), dtype=int)
        
        input = input.to(device)
        input_len = input_len.to(device)
        start_token = start_token.to(device)
        target_E = target_E.to(device)
        
        output = model(input, start_token, input_len, MAX_LEN, True)
        loss = criterion(torch.transpose(output, 1, 2), target_E)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
    


# ================================================================================
# Create dataloader and model
    
dataset, train_loader, valid_loader = get_dataloaders(MAX_LEN, BATCH_SIZE, TRAIN_RATIO)

model = Transformer(
    input_vocab_size=dataset.input_lang.n_words,
    output_vocab_size=dataset.output_lang.n_words,
    embed_dim=512,
    ffn_num_hiddens=1024,
    num_heads=8,
    num_layers=8,
    dropout=0.3
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
epochs = []
train_losses = []
valid_losses = []

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    train_loss = train(train_loader, model, DEVICE, criterion, optimizer)
    valid_loss = validate(train_loader, model, DEVICE, criterion)
    epochs.append(epoch)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    time_spent = asMinutes(time.time() - start_time)
    
    print(f'Epoch[{epoch}/{EPOCHS}]: Train Loss: {train_loss:.4f}. Valid Loss: {valid_loss:.4f}. Time spent:{time_spent}.')
    
print("Training complete! Saving model...")
path = MODEL_PATH + MODEL_NAME + '-' + DATASET_NAME + '.pth'
torch.save(model.state_dict(), path)
print("------------------------------------------------------")



# ================================================================================
# Plot progress

print("Plotting loss graph...")
plt.figure()
plt.plot(epochs, train_losses, label = "Training Loss")
plt.plot(epochs, valid_losses, label = "Validation Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(OUTPUT_PATH + MODEL_NAME + '-' + DATASET_NAME + '.png')



# ================================================================================
# Visualize attention