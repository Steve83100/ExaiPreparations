from io import open
import unicodedata
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler

torch.manual_seed(86)



# ================================================================================
# Dictionary for a language. Contains "word -> index" and "index -> word" mappings

SOS_token = 1
EOS_token = 0 # Make EOS equal 0, so that empty slots are filled with EOS

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "<SOS>", 0: "<EOS>"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # add all words in a sentence
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # add a word to dictionary (index depends on order added, not alphabetical order)
        if word not in self.word2index:
            self.word2index[word] = self.n_words # add mapping: word -> n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word # add mapping: n_words -> word
            self.n_words += 1
        else:
            self.word2count[word] += 1



# ================================================================================
# Read data file, split into pairs, and normalize characters (use lowercase ASCII)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    lines = open('/home/hqdeng7/syang/Datasets/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # make empty language dictionaries
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



# ================================================================================
# Filter sentences by length and content

def filterPair(p, max_len, prefixes):
    return (
        (len(p[0].split(' ')) < max_len and len(p[1].split(' ')) < max_len)
        and
        (p[0].startswith(prefixes) or p[1].startswith(prefixes))
        )

def filterPairs(pairs, max_len, prefixes):
    return [pair for pair in pairs if filterPair(pair, max_len, prefixes)]

# By default keeps sentences starting with "i am" "i'm" "he is"...
PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)



# ================================================================================
# Complete Lang building

def buildLang(lang1, lang2, max_len, prefixes = PREFIXES, reverse=False):
    '''
    Prepares language translation data
    
    :param lang1: input language, such as "eng"
    :param lang2: output language, such as "fra"
    :param max_len: length limit of sentences. Will filter out sentences with exceeding length
    :param prefixes: prefix limit of sentences. Will filter out sentences not starting with them
    :param reverse: whether to reverse input and output language
    '''
    
    print("\nReading dataset...")
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Total: %s sentence pairs" % len(pairs))
    
    print("\nFiltering dataset...")
    pairs = filterPairs(pairs, max_len, prefixes)
    print("Remaining: %s sentence pairs" % len(pairs))
    
    print("\nAdding words to dictionary...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Dictionary size:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



# ================================================================================
# Helper functions for creating dataset

def indexesFromSentence(lang, sentence):
    '''Takes in a sentence, returns a list of word indexes.'''
    return [lang.word2index[word] for word in sentence.split(' ')]

# def tensorFromSentence(lang, sentence, device):
#     '''Takes in a sentence, returns a tensor of word indexes, with EOS appended to the end.'''
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# def tensorsFromPair(pair, device):
#     input_tensor = tensorFromSentence(input_lang, pair[0], device)
#     target_tensor = tensorFromSentence(output_lang, pair[1], device)
#     return (input_tensor, target_tensor)



# ================================================================================
# Create dataset for translation

class EnFrDataset(Dataset):
    """English to French dataset, contains: (en, en_len, SOS+fr, fr+EOS)."""
    def __init__(self, max_len):
        super().__init__()
        
        # Load dictionary and pairs
        self.input_lang, self.output_lang, self.pairs = buildLang('eng', 'fra', max_len)
        self.size = len(self.pairs)
        
        # Create empty sequence vectors (0 means EOS)
        self.input_ids = (np.zeros((self.size, max_len), dtype=np.int32))
        self.input_lens = (np.zeros((self.size), dtype=np.int32))
        self.target_ids_SOS = (np.zeros((self.size, max_len), dtype=np.int32))
        self.target_ids_EOS = (np.zeros((self.size, max_len), dtype=np.int32))

        # Fill each sequence vector with a sentence of word indexes
        for idx, (inp, tgt) in enumerate(self.pairs):
            inp_ids = indexesFromSentence(self.input_lang, inp)
            tgt_ids = indexesFromSentence(self.output_lang, tgt)
            self.input_ids[idx, :len(inp_ids)] = inp_ids
            self.input_lens[idx] = len(inp_ids) # For input sequence, also record length
            self.target_ids_SOS[idx, :(len(tgt_ids)+1)] = [SOS_token] + tgt_ids # For target sequence, append SOS
            self.target_ids_EOS[idx, :(len(tgt_ids)+1)] = tgt_ids + [EOS_token] # EOS not necessary, as it's initialized with 0
            
    def __getitem__(self, index):
        input_id = torch.tensor(self.input_ids[index], dtype=torch.long)
        input_len = self.input_lens[index]
        target_id_SOS = torch.tensor(self.target_ids_SOS[index], dtype=torch.long)
        target_id_EOS = torch.tensor(self.target_ids_EOS[index], dtype=torch.long)
        return input_id, input_len, target_id_SOS, target_id_EOS

    def __len__(self):
        return self.size
    
    def decode(self, index, En = True):
        """Given index, return the word in dictionary"""
        if En:
            return self.input_lang.index2word[index]
        else:
            return self.output_lang.index2word[index]



# ================================================================================
# Create training and validating dataloaders

def get_dataloaders(max_len, batch_size, train_ratio):
    dataset = EnFrDataset(max_len)
    
    # Split dataset into train and validation
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size # Remaining size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle = False, batch_size = batch_size)
    
    return dataset, train_loader, valid_loader



# ================================================================================
# Testing

if __name__ == "__main__":
    print("Testing...")
    
    # input_lang, output_lang, pairs = buildLang('eng', 'fra', MAX_LEN)
    # print("\nSample word:", random.choice(list(input_lang.word2index.items())))
    # print("Sample word:", random.choice(list(output_lang.word2index.items())))
    # print("Sample pair:", random.choice(pairs))
    
    dataset = EnFrDataset(20)
    input, input_len, target_S, target_E = dataset[0]
    print("\nSample data:")
    print([dataset.decode(index.item()) for index in input])
    print(input_len)
    print([dataset.decode(index.item(), En=False) for index in target_S])
    print([dataset.decode(index.item(), En=False) for index in target_E])