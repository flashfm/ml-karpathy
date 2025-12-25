# Shared functions

import sys, os, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import get_file

def get_words():
    all_names = get_file("names.txt", "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
    words = all_names.splitlines()
    return words

def get_dictionaries(words):
    chars = sorted(list(set("".join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi["."] = 0
    itos = {v:k for k,v in stoi.items()}
    return stoi, itos

def build_dataset(words, block_size, stoi):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)