def get_stats(ids: list[int]):
  """
  Calculate frequences of pairs of numbers in the array.

  Example:
    In [0, 1, 2, 0, 1] the result will have (0, 1) -> 2 and (2, 0) -> 1
  
  Args:
    List of numbers.

  Returns:
    Dictionary where keys are pairs and values are frequency.
  """
  result = {}
  for i in range(len(ids)-1):
    pair = (ids[i], ids[i+1])
    result[pair] = result.get(pair, 0) + 1
  return result

def merge(ids: list[int], pair, newNum):
  """
  Replaces all occurrences of the pair of numbers to the new number.

  Returns:
    New array where pair is replaced by newNum.
  """
  result = []
  i = 0
  while i < len(ids):
    if i<len(ids)-1 and (ids[i], ids[i+1]) == pair:
      result.append(newNum)
      i += 2
    else:
      result.append(ids[i])
      i += 1
  return result

def train(ids: list[int], target_vocab_size):
  """
  Continuolsy replace most frequent pair of numbers in the array by a "next" number starting 256 until it hits target_vocab_size.
  Creates a "trained" mapping between pairs and new codes.

  Example:
    Array [0, 1, 2, 0, 1] will be replaced to [256, 2, 256] and so on (since [0, 1] is the most frequent pair).

  Returns:
    Modified array and mapping "pair of numbers -> new number".
  """
  merges = {}
  next = 256
  while next < target_vocab_size:
    stats = get_stats(ids)
    pair = max(stats, key = stats.get)
    merges[pair] = next
    ids = merge(ids, pair, next)
    next += 1
  return ids, merges

def create_vocab(merges):
  """
  Converts a mapping "pair of numbers -> new number" to an array of arrays, where array at position N represents unfolded sequence encoded by N.
  Assumes that the "merges" parameter first item maps to 256.
  Fills first indexes 0..255 by those numbers itself.
  Can be used to decode (decompress).

  Example:
    For {[0,1] -> 256, [256, 2] -> 257} returns [ [0], [1], ..., [255], [0, 1], [0, 1, 2] ]
    
  """
  vocab = [[i] for i in range(256)]
  for (p0, p1), _ in merges.items():
    vocab.append(vocab[p0] + vocab[p1]) # + is array concat
  return vocab

def unicode_to_ids(text):
  """
  Converts unicode representation of string to array of numbers, where every number is a byte converted to int.
  """
  return list(map(int, text.encode("utf-8")))

def encode(tokens, merges):
  """
  Using pre-trained mapping between pairs and new codes in the "merges" dictionary,
  continously finds a pair in the array of tokens and replaces by target number whichever is smaller.
  The process stops when we could not find any pair in the "merges".

  Returns:
    Array encoded using pre-trained merges.
  """
  while True:
    stats = get_stats(tokens) # get all pairs in the tokens
    if not stats:
      break
    pair = min(stats, key = lambda p: merges.get(p, float("inf"))) # a pair from stats, that has minimum value in merges
    if pair not in merges:
      # pair is not in merges - nothing to do anymore
      break
    tokens = merge(tokens, pair, merges[pair])
  return tokens

def decode(ids, vocab):
  """
  Using vocabulary mapping (see create_vocab()) "decompresses" array of numbers into a string.
  Returns:
    Text string.
  """
  decompressed = []
  for n in ids:
    decompressed += vocab[n]
  return bytearray(decompressed).decode("utf-8", errors = "replace")

def bpe_test():
  # First train our "merges" using the following text.

  train_text = """
  It was in July, 1805, and the speaker was the wellknown Anna Pavlovna Scherer, maid of honor and
  favorite of the Empress Marya Fedorovna. With these
  words she greeted Prince Vasili Kuragin, a man of high
  rank and importance, who was the first to arrive at her
  reception. Anna Pavlovna had had a cough for some days.
  She was, as she said, suffering from la grippe; grippe
  being then a new word in St. Petersburg, used only by the
  elite.
  """
  tokens = unicode_to_ids(train_text)
  _, merges = train(tokens, 276)

  # Now let's encode some other text using those "merges" mapping.

  validation_text = """
  ‘Heavens! what a virulent attack!’ replied the prince,
  not in the least disconcerted by this reception. He had just
  entered, wearing an embroidered court uniform, knee
  breeches, and shoes, and had stars on his breast and a
  serene expression on his flat face. He spoke in that refined
  French in which our grandfathers not only spoke but
  thought, and with the gentle, patronizing intonation
  natural to a man of importance who had grown old in
  society and at court. He went up to Anna Pavlovna, kissed
  her hand, presenting to her his bald, scented, and shining
  head, and complacently seated himself on the sofa.
  """
  ids = encode(unicode_to_ids(validation_text), merges)

  # Decode and test that it equals to the original

  vocab = create_vocab(merges)
  text2 = decode(ids, vocab)
  print("Original and decoded texts are equal" if validation_text == text2 else "Error!")

bpe_test()