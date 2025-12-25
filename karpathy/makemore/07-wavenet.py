# Improving our model.
# Additionally to Bengio paper, we'll use paper "WaveNet: A Generative Model for Raw Audio" by Aaron van den Oord et al.

import torch, torch.nn.functional as F, matplotlib.pyplot as plt, random
from mmshared import *

# Performance log:
# Original (context: 3, hidden neurons: 200, params: 12K): train 2.058, val 2.105
# Increase context (context: 8, params: 22K): train 1.918, val 2.027
# Introduced FlattenConsecutive: train 1.941, val 2.029
# Fix bug in batch norm: train 1.912, val 2.022
# Scale up (n_embed 24, n_hidden 123): train 1.769 val 1.993

# region Layer classes

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
                self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
     
    def __init__(self, dim, eps = 1e-5, momentum = 0.1): 
        self.eps = eps
        self.momemtum = momentum
        self.training = True
        # parameters (to be trained)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (running mean and variance (std^2), using "momentum update")
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # forward pass
        if self.training:
            dim = 0 if x.ndim==2 else (0,1) if x.ndim==3 else -1
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momemtum) * self.running_mean + self.momemtum * xmean
                self.running_var = (1 - self.momemtum) * self.running_var + self.momemtum * xvar

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
class Embedding:
    
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
    
class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)

        # if it's (B, 1, Z) then we actually want to remove that 1 and have (B, Z)
        if x.shape[1]==1:
            x = x.squeeze(1)

        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# endregion

random.seed(42)
torch.manual_seed(42)

words = get_words()
random.shuffle(words)

stoi, itos = get_dictionaries(words)

vocab_size = len(itos)
block_size = 8
n_embed = 24
n_hidden = 123

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
Xte, Yte = build_dataset(words[n2:], block_size, stoi)

# We replace the following with the Embedding and Flatten layers:
# C = torch.randn((vocab_size, n_embed))
# And later emb = C[Xb], x = emb.view(emb.shape[0], -1)

# Also instead of array of layers, we use Sequential class, and instance is called "model"

model = Sequential([
    Embedding(vocab_size, n_embed),
    # FlattenConsecutive(block_size) is the same is Flatten()
    FlattenConsecutive(2), Linear(n_embed * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, vocab_size)
])

with torch.no_grad():
    # make last layer not confident (values are low)
    lastLayer = model.layers[-1]
    lastLayer.weight *= 0.1

parameters = model.parameters()

print("parameters:", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

for layer in model.layers:
    layer.training = False

@torch.no_grad()
def split_loss(split):
    x,y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte)
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss("train")
split_loss("val")

def sample():
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix==0:
            break
    print(''.join(itos[i] for i in out))

sample()

def print_shapes():
    for layer in model.layers:
        print(layer.__class__.__name__, ":", tuple(layer.out.shape))

# Let's draw lossi differently: draw mean of every 1000 elements
# torch.arange(10) = [0,1,...,9]
# torch.arange(10).view(-1, 5) = [0,...,5], [6,...,9]
# torch.tensor(lossi).view(-1, 1000).mean(1) - is 200 by 1 (mean of every 1000 elements)

# plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
# plt.show()

# In Bengio paper we connect input layer into second layer and so kinda squashing it fast, in a single step.
# In WaveNet paper, symbols are "slowly" added to multiple layers.
# So pair of symbols are added into "bigram" representation, then two bigrams combined again, etc.
# I.e. progressive fusion using a hierarchical scheme:
#        *
#     /   |   
#    *    *
#   /|   /|
#  * *  * *

# let's see what do we have so far in case of 4 examples:
# ix = torch.randint(0, Xtr.shape[0], (4,))
# Xb, Yb = Xtr[ix], Ytr[ix]
# logits = model(Xb)
# Embedding layer output shape is [4, 8, 10]
# Flattening layer output shape is [4, 80]
# First linear layer: torch.randn(4, 80) @ torch.randn(80, 200) + torch.randn(200)
# as turns out, first multiplier can have shape not only (4, 80) but also (4, x, 80) or (4, x, ..., y, 80)
# only the last dimension is used for multiplication!

# if our X is [1 2 3 4 5 6 7 8] we multiplied previously, now we want to split it into pairs (1 2) (3 4) (5 6) (7 8)
# so we want: torch.randn(4, 4, 20) @ torch.randn(20, 200) + torch.randn(200)
# because embedding size for each pair is now 20 and we have 4 pairs

# So we want the Flatten layer to return (4, 4, 20)
# If emb = torch.randn(4, 8, 10)
# torch.cat([emb[:, ::2, :], emb[:, 1::2, :]], dim=2) would do the trick (a::b is a start and b delta), but
# emb.view(4, 4, 20) gives us the same (4, 4, 20) matrix.
# So we create FlattenConsecutive() layer for that and use 3 times [FlattenConsecutive(2), linear, tanh and batch norm]

# After FlattenConsecutive introduced (32 input examples) and with 68 nodes in hidden layer we have a hierarchical picture like in paper:
# print_shapes()

# Embedding : (32, 8, 10)
# FlattenConsecutive : (32, 4, 20)
# Linear : (32, 4, 68)
# BatchNorm1d : (32, 4, 68)
# Tanh : (32, 4, 68)
# FlattenConsecutive : (32, 2, 136)
# Linear : (32, 2, 68)
# BatchNorm1d : (32, 2, 68)
# Tanh : (32, 2, 68)
# FlattenConsecutive : (32, 136)
# Linear : (32, 68)
# BatchNorm1d : (32, 68)
# Tanh : (32, 68)
# Linear : (32, 27)

# However, the BatchNorm1d now does not do what it supposed to.
# e = torch.randn(32, 4, 68)
# emean = e.mean(0, keepdim=True) # 1, 4, 68
# evar = e.var(0, keepdim=True) # 1, 4, 68
# ehat = (e - emean) / torch.sqrt(evar + 1e-5) # 32, 4, 68
# running_mean.shape # [1, 4, 68]
# So means are calculated not via all batch.
# This can be fixed: e.mean((0,1), keepdim=True) # 1, 1, 68 <-- fixed now
# So BatchNorm1d was fixed.

# Paper says about convolutional layer. It allows to use batch in parallel in CUDA instead of a loop.