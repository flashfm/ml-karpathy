# Check bpe.py first.
# Then check GPT-2 encoder: https://github.com/openai/gpt-2.
# This example shows how GPT-2 data structures are converted to what we had in bpe.py.

import bpe
import requests, os, json
import regex as re

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def encode_with_pat(text, merges, encoder):
  ids = []
  for s in re.findall(pat, text):
    tokens = [encoder[byte_encoder[b]] for b in s.encode("utf-8")]
    ids.extend(encode(tokens, merges))
  return ids

def convert_bpe_merges(bpe_merges, encoder):
  merges = {}
  for a, b in bpe_merges:
    pair = (encoder[a], encoder[b])
    id = encoder[a+b]
    merges[pair] = id
  return merges

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

def getfile(filename, url, is_json = False):
  if not os.path.exists(filename):
      response = requests.get(url)
      with open(filename, 'wb') as f:
          f.write(response.content)

  with open(filename, 'r', encoding='utf-8') as f:
    if (is_json):
      result = json.load(f)
    else:
      result = f.read()

  return result

encoder = getfile("/content/encoder.json", "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json", is_json = True)

bpe_data = getfile("/content/vocab.bpe", "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe")
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

merges = convert_bpe_merges(bpe_merges, encoder)

ids = encode_with_pat(validation_text, merges, encoder)

print(len(ids))
print(ids)
print("Now compare it to Tiktoken app: https://tiktokenizer.vercel.app/?model=gpt2")
