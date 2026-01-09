from __future__ import unicode_literals, print_function, division
from EngFra import *
from Transformer import *
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler

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
# Create training and validating data

def indexesFromSentence(lang, sentence):
    '''Takes in a sentence, returns a list of word indexes.'''
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    '''Takes in a sentence, returns a tensor of word indexes, with EOS appended to the end.'''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloaders():
    # Load dictionary and pairs
    input_lang, output_lang, pairs = prepareData('eng', 'fra')

    # Create empty one-hot vectors
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # Fill one-hot vectors based on indexes
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    # Merge input and target to form a complete dataset
    dataset = TensorDataset(torch.LongTensor(input_ids).to(DEVICE), torch.LongTensor(target_ids).to(DEVICE))
    
    # Split dataset into train and validation
    train_size = int(TRAIN_RATIO * len(dataset))
    valid_size = len(dataset) - train_size # Remaining size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, shuffle = False, batch_size = BATCH_SIZE)
    
    return input_lang, output_lang, train_loader, valid_loader



# ================================================================================
# Instantiate and train model

input_lang, output_lang, train_loader, valid_loader = get_dataloaders()
data, label = train_loader.next()
print(data.shape)