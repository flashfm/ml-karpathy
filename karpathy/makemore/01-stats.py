# We are building a generator of new people names, that is trained on existing names.
# Bigram character level model: calculates probabilities for symbol B to go after symbol A.

import torch, matplotlib.pyplot as plt
from mmshared import *

# random generator
g = torch.Generator().manual_seed(2147483647)

words = get_words()
stoi, itos = get_dictionaries(words)

# Solve the task using "counting method"
# N[a, b] is number of times when symbol with index 'a' is followed with symbol with index 'b'
# '.' is used to mark start and end of a word.

N = torch.zeros((27, 27), dtype = torch.int32)

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1

# P are probabilities that 'b' follows 'a'
# N+1 is done to avoid 0 probability, since if it's zero then log is -inf, which is bad for our cost function (see below).
# It's called "model smoothing".

P = (N+1).float()
P /= P.sum(1, keepdim = True)

def draw():
  plt.figure(figsize=(16,16))
  plt.imshow(N, cmap = "Blues")
  for i in range(27):
    for j in range(27):
      chstr = itos[i] + itos[j]
      plt.text(j, i, chstr, ha = "center", va = "bottom", color = "gray")
      plt.text(j, i, N[i, j].item(), ha = "center", va = "top", color = "gray")
  plt.axis("off")
  plt.show()

def test():
  for i in range(5):
    ix = 0
    result = ""
    while True:
      p = P[ix]
      ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
      result += itos[ix]
      if ix == 0:
        break
    print(result)

test()

# Quality of model is defined by P.
# We want product of all P[i,j] (called likelihood) to be maximal.
# Each P[i,j] is < 1, so the product will be very small number.
# So we better work with "log likelyhood". Log for numbers 0..1 is negative, so we should use "negative log likelyhood".
# Note that log(a*b) = log(a) + log(b), so we should normalize (average) by n (number of pairs).
# Average negative loss likelyhood is our cost function!
# When it is smaller, then product of P[i,j] is higher.

def get_loss():
  log_likelyhood = 0
  n = 0
  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      prob = P[ix1, ix2]
      logprob = torch.log(prob)
      log_likelyhood += logprob
      n += 1
  loss = -log_likelyhood/n
  return loss

# prints ~ 2.45
print(get_loss())