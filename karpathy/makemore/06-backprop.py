# Check 05-layers.py first.
# Here we implement backprop by ourselves.

import torch, torch.nn.functional as F, matplotlib.pyplot as plt, random
from mmshared import *

g = torch.Generator().manual_seed(2147483647)
words = get_words()
stoi, itos = get_dictionaries(words)

# Function to compare gradients that we calculated ourselves with PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

# Boilerplate from 04-optmizations.py

block_size = 3
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)

vocab_size = len(itos)
n_embed = 10
n_hidden = 64

# Layers
# We aren't using all zeros for the following layers to prevent masking problems with backward pass.

C = torch.randn((vocab_size, n_embed), generator=g)
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * (5/3)/((n_embed * block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
bngain = torch.randn((1, n_hidden)) * 0.1 + 0.1
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, W2, b2, bngain, bnbias]
print("Parameters:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

# Batch
n = batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# Forward
# Create more variables so we can get derivatives later.

emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

hprebn = embcat @ W1 + b1

bnmeani = 1/n * hprebn.sum(0, keepdim=True) # mean
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True) # it's `1/n * ...` in the paper for training and `1/(n-1)` (Bessel's correction) for inference, we use Bessel's corr. always
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)

logits = h @ W2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward

for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn, embcat, emb]:
    t.retain_grad()
loss.backward()

# Let's create our own backward pass!
# We'll compare it to PyTorch.

# Chain rule:
# If we have a chain: x → u → v → y, where u = f(x), v = g(u), y = h(v)
# Then dy/dx = (dy/dv) * (dv/du) * (du/dx)

# Our chain is x -> ... -> loss
# Our goal is d(loss) / d(x).
# On every step we are going to calculate d(loss) / d(intermediate), until we get to the goal.
# Variable keeping d(loss)/d(intermediate) will be called as dintermediate.

# Example: dprobs = d(loss)/d(probs) = d(loss)/d(logprobs) * d(logprobs)/d(probs) = dlogprobs * d(logprobs)/d(probs)
# where d(logprobs)/d(probs) is derivative of logprobs function with respect of its input probs.

# Step 1.
# Calculate derivative of `loss`, i.e. derivative of `-logprobs[range(n), Yb].mean()` by every "variable". E.g. every changing weight/parameter.

# dlogprobs is derivative of the loss with respect of all elements of logprobs.
# logprobs.shape is [32, 27] so the dlogprobs will be the same.

# logprobs[range(n), Yb] means - take every row, and for each row take column index from Yb vector
# I.e. we select a vector of logprops corresponding to labels in Yv. Shape of logprobs[range(n), Yb] is 32.

# loss = -mean(...) i.e. -(a + b + c + ...) / n. So dloss by da is -1/n.
# Derivative of other elements of logprobs is 0 because they do not participate in the loss calculation.

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n

cmp('logprobs', dlogprobs, logprobs)

# Step 2.
# dprobs = d(loss)/d(probs) = d(loss)/d(logprobs) * d(logprobs)/d(probs) = dlogprobs * d(logprobs)/d(probs)
# logprobs = ln (probs), (ln x)' = 1/x, so d(logprobs)/d(probs) = 1/probs
# by chain rule we multiply to previous

dprobs = (1.0 / probs) * dlogprobs

cmp('probs', dprobs, probs)

# Step 3
# d(loss) / d(counts_sum_inv) = d(loss) / d(probs) * d(probs) / d(counts_sum_inv) = dprobs * d(probs) / d(counts_sum_inv)
# However, `probs = counts * counts_sum_inv` is not a simple multiplication (if it would then derivative would be just `counts`).
# counts.shape is [32,27], counts_sum_inv.shape is [32,1], and this is not a dot-product (@) but element-wise multiplication.
# So PyTorch broadcasts (copies) single column of `counts_sum_inv` into 27 columns, then multiplies element-wise.
# Example:
# c = a * b
# a[3,3] * b[3,1]
# a11*b1 a12*b1 a13*b1
# a21*b2 a22*b2 a23*b2
# a31*b3 a32*b3 a33*b3

# Chain rule in multiple nodes:
#   /-> y1 -\
# x --> y2  -> z
#   \-> y3 -/
# y1 = f1(x)
# y2 = f2(x)
# y3 = f3(x)
# Z = g(y1​,y2​,y3​) - multivariable function.
# Chain rule: d(Z)/dx = d(Z)/dy1 * dy1/dx   +   d(Z)/dy2 * dy2/dx   +   d(Z)/dy3 * dy3/dx

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts = counts_sum_inv * dprobs

dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

cmp('counts_sum', dcounts_sum, counts_sum)

# a11 a12 a13 ->  b1  (= a11 + a12 + a13)
# a21 a22 a23 ->  b2  (= a21 + a22 + a23)
# a31 a32 a33 ->  b3  (= a31 + a32 + a33)
# Replicate (broadcast) dcounts_sum over counts
dcounts = dcounts + torch.ones_like(counts) * dcounts_sum 

cmp('counts', dcounts, counts)

# If f = norm_logits.exp(), then f' = norm_logits.exp() too, because e'(x) = e(x), already calculated in `counts`

dnorm_logits = counts * dcounts

cmp('norm_logits', dnorm_logits, norm_logits)

# `norm_logits = logits - logit_maxes` here we have logit_maxes (32,1) broadcasted to (32,27) again
dlogits = dnorm_logits.clone()

dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)

cmp('logit_maxes', dlogit_maxes, logit_maxes)

# Partial derivative in matrix will be 1 in places where max was sitting, and 0 otherwise.
dlogits = dlogits + F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

cmp('logits', dlogits, logits)

# Find derivative of a@b+c on paper w.r.t. a, you will see that 

dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)

# tanh' = 1 - tanh^2
dhpreact = (1 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)

cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)

dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = -0.5 * ((bnvar + 1e-5)**-1.5) * dbnvar_inv

cmp('bnvar', dbnvar, bnvar)

dbndiff2 = 1/(n-1) * torch.ones_like(bndiff2) * dbnvar

cmp('bndiff2', dbndiff2, bndiff2)

dbndiff += 2 * bndiff * dbndiff2

cmp('bndiff', dbndiff, bndiff)

# emb = C[Xb]
# embcat = emb.view(emb.shape[0], -1)

dhprebn = dbndiff.clone()

# `bndiff = hprebn - bnmeani` - broadcasting here, so need sum
dbnmeani = -dbndiff.sum(0, keepdim=True)

cmp('bnmeani', dbnmeani, bnmeani)

dhprebn += 1/n * torch.ones_like(hprebn) * dbnmeani

cmp('hprebn', dhprebn, hprebn)

dembcat = dhprebn @ W1.T

cmp('embcat', dembcat, embcat)

dW1 = embcat.T @ dhprebn

cmp('W1', dW1, W1)

db1 = dhprebn.sum(0)

#cmp('b1', db1, b1)

demb = dembcat.view(emb.shape)

cmp('emb', demb, emb)

# emb = C[Xb] - plucking out rows, have to use for-loop to get back to propagate gradient
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k,j]
        dC[ix] += demb[k,j]

cmp('C', dC, C)

######
# We can simplify loss calculation and so make gradients calculation more efficient.

# loss = -log Py
# Pi = e^Li / sum(e^Lj)
# dloss / dLi = d/dLi (-log e^Li...)

# Do calculation on paper...
# dloss / dLi = ... = if i!=y then Pi, otherwise Pi-1

###
# Let's start with dlogits:

dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n

cmp('logits', dlogits, logits)

# Let's visualize first row of logits and dlogits
print("logits: ", F.softmax(logits, 1)[0])

# Note the -0.96 value: it is ~1 and all other values are ~0
print("dlogits:", dlogits[0] * n)

# Sum of graditents is 0
print("dlogits sum", dlogits[0].sum())

# So dlogits pulls "correct" value up and "wrong" values down.

# Also for all examples:
# plt.figure(figsize=(8,8))
# plt.imshow(dlogits.detach(), cmap='gray')
# plt.show()

###
# Now, lets calculate hpreact more efficiently:

# forward pass
# bnmeani = 1/n * hprebn.sum(0, keepdim=True) # mean
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True) # it's `1/n * ...` in the paper for training and `1/(n-1)` (Bessel's correction) for inference, we use Bessel's corr. always
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

hpreact_fast = bngain * ((hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt((hprebn.var(0, keepdim=True, unbiased=True) + 1e-5))) + bnbias

print("hpreact_fast max diff: ", (hpreact - hpreact_fast).abs().max())

# backward pass
# let's calculate dhprebn (given dhpreact)
# Use paper "Batch normalization: accelerating deep network training by reducing internal covariate shift" by Sergey Ioffe at al
# to get formulas of mu, sigma-squared, X-Ith with hat, Mu-Ith etc. and calculate derivatives on paper.

dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))

cmp('hprebn', dhprebn, hprebn)

# Now we can use those derivatives to update the values instead of calling loss.backward():
# for p, grad in zip(parameters, grads):
#   p.data += -lr * grad