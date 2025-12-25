# Check 04-optimizations.py first.
# In this file we refactor it into layers, add more layers and do visualizations to analyze how weights are changing.

import torch, torch.nn.functional as F, matplotlib.pyplot as plt, random
from mmshared import *

g = torch.Generator().manual_seed(2147483647)
random.seed(42)

# For W1, b1 or W2, b2
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
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
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
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
    
words = get_words()
random.shuffle(words)

stoi, itos = get_dictionaries(words)

vocab_size = len(itos)
block_size = 3
n_embed = 10
n_hidden = 100

C = torch.randn((vocab_size, n_embed), generator=g)

# First, use a version without BatchNorm1d

# layers = [
#     Linear(n_embed * block_size, n_hidden), Tanh(),
#     Linear(            n_hidden, n_hidden), Tanh(),
#     Linear(            n_hidden, n_hidden), Tanh(),
#     Linear(            n_hidden, n_hidden), Tanh(),
#     Linear(            n_hidden, n_hidden), Tanh(),
#     Linear(            n_hidden, vocab_size) # "softmax" layer
# ]

layers = [
    Linear(n_embed * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(            n_hidden, vocab_size), BatchNorm1d(vocab_size) # "softmax" layer
]

with torch.no_grad():
    # make last layer not confident (values are low)
    lastLayer = layers[-1]
    if isinstance(lastLayer, BatchNorm1d):
        lastLayer.gamma *= 0.1
    else:
        lastLayer.weight *= 0.1

    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3 # see 04-optimizations.py to understand what 5/3 means

parameters = [C] + [p for layer in layers for p in layer.parameters()]

print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []
ud = []

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)

for i in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad() # Remove after debugging
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < max_steps/2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])

    if i==1000:
        break # Remove after debugging

def visualize(title, func):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = func(layer)
            saturation = (t.abs() > 0.97).float().mean() * 100
            print('layer %d (%10s): mean %+f, std %+f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), saturation))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title(f'{title} distribution')
    plt.show()

# Experiment with weight multiplier: put something except 5/3 to see how saturation is changed.
# With 5/3 the saturation is about 5% which is ok.

# visualize('activation', lambda l: l.out)

# With multipliers different from 5/3 the gradients also differ per layer.
# With 5/3 they are roughly the same.

# visualize('gradient', lambda l: l.out.grad)

# With small value like 0.5 the activations shrink to 0 and gradients different on each layer - start at 0 and expanding.
# With high value like 3 activations -1 and +1 increases, gradients become different.

# If we remove Tanh layer, we see that the proper multiplier would be 1 instead of 5/3.

# We want gradients to have similar distribution layer over layer - not fading away, not exploding. Because we can have 50+ layers.

# If we remove Tanh, all other linear layers can be represented with just one large linear layer.
# Tanh allow model represent non-linear functions. Linear is scaling and rotating. Non-linear is bending.

def visualize_params():
    plt.figure(figsize=(20, 4))
    legends = []
    for i,w in enumerate(parameters):
        g = w.grad
        if w.ndim==2: # weights
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(w.shape), g.mean(), g.std(), g.std() / w.std()))
            hy, hx = torch.histogram(g, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} ({tuple(w.shape)})')
    plt.legend(legends)
    plt.title('gradients of weights')
    plt.show()

# visualize_params()

# If we visualize gradient on parameters, we'll see that last layer looks different - gradients are too big there after 1 pass.
# After 1000 passes it is kinda stabilized - all gradients start look more or less the same, but the last layer still a bit different.

def visualize_ud():
    plt.figure(figsize=(20, 4))
    legends = []
    for i,w in enumerate(parameters):
        if w.ndim==2: # weights
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
    plt.plot([0, len(ud)], [-3, -3], 'k') # ratios should be about 1e-3 (baseline)
    plt.legend(legends)
    plt.title('update ratios')
    plt.show()

# Finally, we can visualize updates as "update ratios" where we see what was the ratio of update (lr * grad) to weight (we take std) and log scale.

visualize_ud()

# We want to see update ratios to be about 1e-3.
# If its too low, then training goes too slow.
# Last softmax layer is an exception.
# E.g.
# - if we set LR to 0.001 then we see how update ratios change to 1e-5 or so.
# - if we don't use `/ fan_in**0.5` initial weight factor, then we see how update ratios will be very different for all layers.

# After BatchNorm1d added, we see how update ratios are set to about 1e-3. Other visualizations are always OK.
# If we set initial weight multiplier to 0.2, we'll see that while other charts are OK, the update ratios will be increased to about 1e-1.
# If we set it to 5.0, we'll see that update ratios become lower.

# If we remove `/ fan_in**0.5` and weights multiplier, keeping only batch norm, then update ratios will be around 1e-4.

# Low update ratios can be "fixed" by higher update rate.
# E.g. 10x (LR=1) to get update ratios about 1e-3. 