# Check stats.py first.
# Same problem, but we'll use a neural network instead of just statistics

import torch, torch.nn.functional as F, matplotlib.pyplot as plt
from mmshared import *

# random generator
g = torch.Generator().manual_seed(2147483647)

words = get_words()
stoi, itos = get_dictionaries(words)

# Let's create a training set of bigrams (x, y). Meaning: x -> y.
# We'll use vectors xs and ys.
# xs is array of inputs.
# ys is array of outputs.
# (but indexes, not chars)
# Example: in case of .emma.:
# xs: . e m m a
# xy: e m m a .

def create_training_set(words):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    return torch.tensor(xs), torch.tensor(ys)

xs, ys = create_training_set(words)

num = xs.nelement()
print("number of examples:", num)

# Initialize the neural network

# We have 27 letters, each letter is represented not by a number, but with one-hot vector, i.e. [0,..,1,..,0] where 1 is on the position of that number.
# Neural network input should be one-hot vector since internally it multiplies numbers.

xenc = F.one_hot(xs, num_classes = 27).float()

# xenc is multiple chars (inputs) encoded to one-hot.
# It is a matrix where each row is one-hot vector of appropriate input example (number).
# Neural net works with multiple input examples simultaneously packed into a matrix.

print(xenc.shape)
plt.imshow(xenc)
# plt.show()


# define initial random weights
W = torch.randn((27, 27), generator = g, requires_grad = True)

# Gradient descent

for k in range(100):

  # Forward pass

  # Our layer has no bias and no activation function, so we just multiply input by weights.
  # Similarly to stats.py, we want output to be an analogue of counts and then we can calculate probabilities.
  # Neural network does not work with int counts as they may be large, so we consider it will output "log counts" called "logits".
  # If we exponentiate them using exp() then we get equivalent of counts (matrix N).

  logits = xenc @ W
  counts = logits.exp()

  # Similarly how we calculated P, we calc probabilities here.
  # Every row is a row of probabilities of one of the 27 letters.
  # I.e. you may see it as like every input one-hot vector represent 100% probability (1 is 100%) of specific letter,
  # and now we have non-100% for every letter.
  # Such exp() + normalizing is called "soft-max function for logits".
  # We convert [1.3, 5.1, 2.2, 0.7, 1.1] -- softmax -> [0.02, 0.9, 0.05, 0.01, 0.02]. Higher the number - higher the prob, sum is 1.

  probs = counts / counts.sum(1, keepdim = True)

  # Btw, remember that we have multiple examples in the matrix, each row is one example.

  # Let's try to tune our W to improve probs by reducing our loss function (defined in stats.py).

  # Calculate loss (average negative log likelyhood)

  # For-loop form, we won't use it since it is slow.
  # nlls = torch.zeros(num)
  # for i in range(num):
  #   x = xs[i].item()
  #   y = ys[i].item()
  #   print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x}, {y})')
  #   print('input to the neural net:', x)
  #   print('output probabilities from the neural net:', probs[i])
  #   print('label (actual next character):', y)
  #   # probs[i] is probabilities for all chars, so probs[i, y] is probability assigned to y
  #   p = probs[i, y]
  #   nll = -torch.log(p)
  #   nlls[i] = nll
  # # average negative log likelyhood is loss
  # loss = nlls.mean().item()

  # Tensor form.
  # torch.arange(5) = [0, 1, 2, 3, 4]
  loss = -probs[torch.arange(num), ys].log().mean()

  # Regularization.
  # That prevents having some W values too small and some too large. Aka smoothing the probabilities.
  # If we incentivize W values to be around zero, then distribution would be more smooth.
  # It's like we did (N+1) in stats.py.
  # If we would add 1_000_000, then all probabilities would be equal (since values of N are much smaller) - this is super-smooth.
  # We convert W values to all positives via ^2, then take average to have it small (if we'd take sum() it would be too large).
  # Example: (W**2).mean() can be = 2.112 on first run.
  # We also multiply by "regularization strength" equals 0.01.
  # If W values are far from 0, then loss increases and the net will try to make it smaller next time.
  # If we would choose "regularization strength" too big, it would overweight the first part of "loss",
  # so the optimization would be too focused on making W small, not by having probabilities follow examples.
  loss += 0.01 * (W**2).mean()

  print("loss:", loss.item())

  # backward pass

  # set gradient to 0
  W.grad = None
  loss.backward()

  # Update weights
  # Learning rage is big but it works fine on this small model.
  W.data += -50 * W.grad

# At the end we should have W values similar to what we had in P in stats.py
# That's because one-hot vector multiplied by W should give one row of W, and that's the same like we had in P.

# Let's sample
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))