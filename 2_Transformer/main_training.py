from __future__ import unicode_literals, print_function, division
import EngFra
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")



# Prepare language dictionaries and sentence pairs

input_lang, output_lang, pairs = EngFra.prepareData('eng', 'fra')

print("\nSample word:", random.choice(list(input_lang.word2index.items())))
print("Sample word:", random.choice(list(output_lang.word2index.items())))
print("Sample pair:", random.choice(pairs))



# Create training and testing datasets