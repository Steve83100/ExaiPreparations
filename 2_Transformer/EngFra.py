from io import open
import unicodedata
import re
import random



# ================================================================================
# Dictionary for a language. Contains "word -> index" and "index -> word" mappings

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
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

def filterPair(p, max_length, prefixes):
    return (
        (len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length)
        and
        (p[0].startswith(prefixes) or p[1].startswith(prefixes))
        )

def filterPairs(pairs, max_length, prefixes):
    return [pair for pair in pairs if filterPair(pair, max_length, prefixes)]

# By default keeps sentences no longer than 10 words
MAX_LENGTH = 10

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
# Complete preparation

def prepareData(lang1, lang2, max_length = MAX_LENGTH, prefixes = PREFIXES, reverse=False):
    '''
    Prepares language translation data
    
    :param lang1: input language, such as "eng"
    :param lang2: output language, such as "fra"
    :param max_length: length limit of sentences. Will filter out sentences with exceeding length
    :param prefixes: prefix limit of sentences. Will filter out sentences not starting with them
    :param reverse: whether to reverse input and output language
    '''
    
    print("Reading dataset...")
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Total: %s sentence pairs" % len(pairs))
    
    print("\nFiltering dataset...")
    pairs = filterPairs(pairs, max_length, prefixes)
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
# Testing

if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
    print("\nSample word:", random.choice(list(input_lang.word2index.items())))
    print("Sample word:", random.choice(list(output_lang.word2index.items())))
    print("Sample pair:", random.choice(pairs))