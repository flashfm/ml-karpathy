# Check stats.py and neural.py first.
# Same problem, but instead of bi-grams like in neural.py, we introduce embeddings matrix and "context"

import torch, torch.nn.functional as F, matplotlib.pyplot as plt, random
from mmshared import *

g = torch.Generator().manual_seed(2147483647)
random.seed(42)

words = get_words()
random.shuffle(words)

stoi, itos = get_dictionaries(words)

block_size = 3 # context length, each input example is block_size length

# Tip: work on a small batch of words in the beginning: e.g. words[:5]

def build_dataset(words):
    X = [] # inputs (each item is vector with size block_size)
    Y = [] # outputs (each item is "next character")
    for w in words:
        # print(w)
        context = [0] * block_size # [0,0,0] if block_size is 3
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '->', itos[ix])
            context = context[1:] + [ix] # remove first and append last
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# emma
# ... -> e
# ..e -> m
# .em -> m
# emm -> a
# mma -> .

# Number of parameters (neurons etc.) in the model can be very large.
# Much larger than information in the training set.
# So the model will start to completely memorize the training set.
# This is called "overfitting".
# To prevent that, the data set is divided into splits:
# - training (80%) - for backprop
# - dev/validation (10%) - to choose hyperparameters (size of layers, strenght of regularization (not used yet))
# - test (10%) - for evaluation

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Now we need to place each "letter" (we have 27 of them) in the N-dimensional space.
# Tip: take N = 2 at the beginning. We use 10 here by the reasons described at the end.
# So let's have C as tensor with 27 rows and N columns. Each row for every character, and row is N-dimensional vector.
C_size = 10 # 2 in the beginning
C = torch.randn((27, C_size))

# Note:
# We previously represented each of 27 chars as 1-hot vectors [0, ..., 1, ..., 0, ...0]
# Nth char is represented by one-hot vector where 1 is on Nth place.
# Multiplying that vectory by C will give us C[N], because zeros will mask-out C rows.
# So, C can be also considered as weights of 1-layer of neurons without activation function. 

# Pytorch indexing:
# C[5] gives a vector [a,b]
# C[[5,6,7]] gives three vectors (shape is [3,2])
# C[X] gives M vectors (X has M rows), shape is [M, 3, 2].
# X[0] is [0, 0, 0] (for '...'), so C[X] is
# [
# [[a, b], [a, b], [a, b]], - vector for the first '...',
# [[a, b], [a, b], [c, d]], - verctor for the second '..e',
# [[a, b], [c,d], [e,f]], - verctor for the third '.ef',
# and so on
# ]
# e.g X[13, 2] = 1, then C[X][13, 2] == C[1]

# emb = C[X]

W1_input_size = C_size * block_size # C_size is dimension of embedding matrix; block_size is length of input sample
W1_size = 200 # number of neurons in this layer (100 originally)
W1 = torch.randn((W1_input_size, W1_size))
b1 = torch.randn(W1_size) # biases

# "emb @ W1 + b1" won't work since the emp.shape = [M, 3, 2], where M is number of input examples
# emb[:, 0, :].shape is [M, 2]
# concatenation emb[:, 0, :], emb[:, 1, :], emb[:, 2, :] is what we need
# torch.cat(a, 1) concatenates tensor a along dimension with index 1
# so we could build it: a = [], a.append(emb[:, 0, :]), a.append(emb[:, 1, :]), ...
# and there is a simpler way: torch.unbind(emb, 1) returns such "a"

# tensor is always an one-dimensional array (vector) in memory
# see https://blog.ezyang.com/2019/05/pytorch-internals/
# tensor class has a view() method
# example: a = torch.arange(18)
# a is [0, ..., 17] vector
# a.view(9, 2) gives [ [0...8], [9..17] ]
# a.view(3, 3, 2) gives [ [[0,1], [2,3], [4,5]], [[6,7],[8,9],[10,11]], ... ]

# luckily, emb.view(M, 6) gives us what we need!
# and if we use emb.view(-1, 6) then pytorch derives what first dimension should be based on the second one

# h = emb.view(-1, 6) @ W1 + b1
# h = torch.tanh(h)

# Note about "+" operation:
# result of @ operation size is [32, 100] (100 is W1 size)
# b1 size is [100]
# Is addition work correctly?
# Torch will broadcast [32,100] to [100] by converting [100] into [1,100] and then copying vertically to all 32 rows and adding them

W2 = torch.randn((W1_size, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

print("Model size (number of parameters):",  sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

steps = 200000

# How do we choose the learning rate for backprop?
# If rate is 0.0001 or 0.001, the loss is decreased slightly or goes back and forth.
# If rate is 1 or 10 then it is also unstable.
# Let's choose it between 1 and 0.001.
# torch.linspace(0, 1, 1000) gives us 1000 numbers going from 0 to 1
# we'll use learning rage changing from 10^-3 to 1 on each step
# If we then record on each step what was exp of rate and what was loss,
# and then plot the chart, we'll see that 0.1 rate is acceptable

# Uncomment to build learning rate plot:
# lre = torch.linspace(-3, 0, steps)
# lrs = 10**lre
# lrei = []
stepi = []
lossi = []

def forward(X_batch, Y_batch):
    emb = C[X_batch]
    h = emb.view(-1, W1_input_size) @ W1 + b1
    h = torch.tanh(h)

    # now as in the neural.py

    logits = h @ W2 + b2

    # counts = logits.exp()

    # prob = counts / counts.sum(1, keepdim=True)

    # Y is rows of "results" or "labels", i.e. next characters in training set
    # we want to get what probability the nework assigned to the char in Y
    # torch.arange(32) is [0..31]
    # ideally prob[torch.arange(32), Y] is what we want to be an array of ones (1)

    # we want to minimize this loss
    # loss = -prob[torch.arange(32), Y].log().mean()

    # F.cross_entropy() method:
    # - effectively calculate counts, prop and loss
    # - effective in back propagation
    # - fixes issue with large logits values: if logits = torch.tensor([-100, -3, 0, 100]) then probs will be [0,0,0,nan]

    loss = F.cross_entropy(logits, Y_batch)
    return loss

for i in range(steps):

    # Mini-batch construct:
    # We have 228146 examples in X. It is too slow to do forward-back passes on all of them.
    # So we can use a random chunk (batch), sacrificing the gradient quality, to progress faster and run more iterations.
    # torch.randint(0, 5, (32,)) returns vector of 32 numbers from 0 to 4
    # 32 is our mini-batch size

    ix = torch.randint(0, Xtr.shape[0], (32,))
    X_batch = Xtr[ix]
    Y_batch = Ytr[ix]

    # ----
    # FORWARD PASS
    # ----

    loss = forward(X_batch, Y_batch)

    # print(loss)

    # ----
    # BACKWARD PASS
    # ----
    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1

    # Learning rate decay
    if i>steps*2/3:
        lr = 0.01

    # Uncomment to build learning rate plot:
    # lr = lrs[i]
    # lrei.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item()) # we use log10() so the chart does not look as a hockey stick

    for p in parameters:
        p.data += -lr * p.grad

# Draw loss by learning rate plot:
# See at which power the loss is low, choose it for learning rate power (10^x)
# plt.plot(lrei, lossi)

# Draw loss by step plot
# If we see a lot of noise (up-down) in loss, increase batch size
# plt.plot(stepi, lossi)
# plt.show()

# Note: loss will not be 0 even if we overfit (32 examples with 3481 parameter network)
# That's because '...' context gives us 'e' in 'emma', 's' in 'sophia', etc.
# I.e. there will be still some loss.

print("Train loss", forward(Xtr, Ytr))
print("Dev loss", forward(Xdev, Ydev))

# If Train loss ~ Dev loss then we are underfitting, not overfitting, , this means the model (number of parameters) is too small.
# So increase W1 size from 100 to bigger number.

# If W1 size increased but loss does not improve, then a reason can be that C size is small - it's a bottleneck, and so W1 is not used fully.
# Change original size of C from 2 to 10.

# Tip: if we want to see how the model clustered letters, use this function to draw contents of C (when C size is 2)
def draw_embedding():
    plt.figure(figsize=(8,8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white")
    plt.grid(True, "minor")

# draw_embedding()
# plt.show()

def sample():
    out = []
    context = [0] * block_size
    ix = -1
    while ix!=0:
        emb = C[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        props = F.softmax(logits, dim=1)
        ix = torch.multinomial(props, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
    print(''.join(itos[i] for i in out))

for _ in range(20):
    sample()