# Check embeddings.py first

import torch, torch.nn.functional as F, matplotlib.pyplot as plt, random
from mmshared import *

# region This is almost a copy of embeddings.py

g = torch.Generator().manual_seed(2147483647)
random.seed(42)

words = get_words()
random.shuffle(words)

stoi, itos = get_dictionaries(words)
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

vocab_size = len(itos)
n_embed = 10
n_hidden = 200

# Why those tensors have multiplies like 0, 0.01 or 0.2? See at the end

C = torch.randn((vocab_size, n_embed), generator=g)
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * 0.3 # Why 0.3 - see at the end
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

# See at the end why we added those
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print("Parameters:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # create minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1

    # apply batch normalization (details at the end of this program)
    mean = hpreact.mean(0, keepdim=True)
    std = hpreact.std(0, keepdim=True)
    hpreact = (hpreact - mean) / std
    hpreact = bngain * hpreact + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i<100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 1000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

    # Uncomment to see which tanh() neurons do not learn
    # If all black, we can increase multiplier for W1 a bit (from 0.1 to 0.2)
    # plt.figure(figsize=(20,10))
    # plt.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")

    # Unconmment to see histogram of values in h
    # plt.hist(h.view(-1).tolist(), 50)
    # plt.show()

    # break # to break after first batch

# plt.plot(lossi)
# plt.show()

@torch.no_grad() # disable gradients
def split_loss(split):
    x,y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte)
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias # batch norm here as well
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss("train")
split_loss("val")

def sample():
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix==0:
            break
    print(''.join(itos[i] for i in out))

# endregion

# Let's fix initial weights, because they affect initial loss change.

##### 1
# Optimization 1: Fix softmax()

# If we look at loss at the beginning, it is very high: 0/ 200000: 27.8817
# That's because the first "logits" have large and deviant values. E.g. [-2, 36, -10, 5 ...]
# However, since we just start, we would like to have uniform distribution, i.e. all numbers roughly the same.
# That's because on start the model does not know anything about the training set and should "suppose" same outcome for every character.
# And the initial loss will be much lower in that case.
# I.e. our goal is to get initial logits like [0.1, 0.05, -0.1, 0, ...] - all closer to zero.

# Simplest way to achieve that: multiply b2 and W2 by 0 (i.e. start them with all zeros).
# However, it's better to start with W2 multiplied by a non-zero number like 0.01.

##### 2
# Optimization 2: Fix tanh() too saturated

# 2nd problem is the tanh() function.
# tanh(-3) ~ -1, tanh(3) ~ 1.
# And initial hpreact values range from -15 to 15.
# So tanh() produces a lot of values very close to 1 and -1.
# You can visualize it using the following statement in the training loop: plt.hist(h.view(-1).tolist(), 50)
# You can see it as values from previous layer pass-through.

# When we do backprop, activation function gradient is calculated like: gradient = activation_func_derivative * incoming_gradient.
# tanh() formula is: (e^x - e^-x) / (e^x + e^-x) same as (1 - e^-2x) / (1 + e^-2x).
# So its derivative is: (1 - tanh^2).
# So when tanh() value is -1 or 1, it's derivative is 0. Whatever incoming_gradient is, it is zero-ed (vanished).

# We can look at all tanh() values per each input sample on a 2D chart:
# plt.figure(figsize=(20,10))
# plt.imshow(h.abs() > 0.99, cmap="gray", interpolation="nearest")
# Color of each cell: white if True, else black
# If we see a completely white column, this means that for every example, tanh was ~1, meaning it is "dead" neuron which doesn't learn on any example.

# Not only tanh(), but also sigmoid() and ReLu() have such a flat regions with the same consequences.

# Note: if learning rate is too high, the value of gradient multiplied by it can change the W and b so much, so it stops learning
# (will be always in that flat region).

# Solution is again, to multiply W2 and b2 by small numbers.

#####
# How do we select 0.01 or 0.1 for W1 and W2 multipliers?
# If we check this example...
#
# x = torch.randn(1000, 10)
# w = torch.randn(10, 200)
# y = x @ w
# print(x.mean(), x.std())
# print(y.mean(), y.std())
# plt.figure(figsize=(20,5))
# plt.subplot(121)
# plt.hist(x.view(-1).tolist(), 50, density=True)
# plt.subplot(122)
# plt.hist(y.view(-1).tolist(), 50, density=True)
#
# ... we'll see that while means of x and y are about 0, the standard deviation is changed from 1 in x to 3 in y - that's the result of multiplication.
# The charts will also display a Gaussian and std() will be visible there.
# Multiplying 'w' by a number >1 will make std() more, and <1 will make it less. Our goal is to make it 1 like in x.
# How do we calculate exact number?
# We'll refer to a paper "Delving deep into rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by Kaiming He at al.
# It defines that for ReLu, we need to use std = sqrt(2/input_size). 2 is called "gain" and derived from the fact that ReLU vanishes half of input.
# Generic formula for std is "gain / sqrt(fan_mode)".
# For tanh() gain is 5/3 (because it squashes the input) and "fan_mode" is "fan_in" = size of input.

# If we check torch.randn(10000).std() it is 1, if it check (torch.randn(10000) * 0.2).std() - it is 0.2
# So the multiplier is std!
# For W1, (5/3)/((n_embed * block_size)**0.5) is about 0.3.


##### 3
# Optimization 3. Batch normalization.
# Described in paper "Batch normalization: accelerating deep network training by reducing internal covariate shift" by Sergey Ioffe at al.
# It says that we can normalize "hpreact" variable per every batch.
# Normalize means that we want a Guassian distribution with 0 mean and 1 standard deviation.
# For that, calculate mean and standard deviation of hpreact, and then do "hpreact = (hpreact - mean) / std".
# Btw, mean and std are differentiable functions, so there will be no problem with backpropagation.

# However, we want model to allow some deviation, i.e. allow the distribution to be shifted a bit, to avoid keeping it 0-mean, 1-std Guassian always.
# So we introduce some "gain" and "shift" (bias) to be learned and modify the hpreact.
# That's why we introduced bngain (all ones) and bnbias (all zeros) at the code above.

# Batch normalization allows to think less about what "scale" multiplier should W1, W2, and so on have.