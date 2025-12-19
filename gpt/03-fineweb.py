# Using FineWeb Edu sample-10BT input dataset to train our GPT-2 model
# See https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

import os, tiktoken, numpy as np, multiprocessing as mp
from tqdm import tqdm  # progress bar
from datasets import load_dataset

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# load file
# fw is list of rows
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(row):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(row["text"]))
    tokens_np = np.array(tokens)

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "tokens are not within 0..2**16 range"

    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(shard_index, tokens_np):
    split = "val" if shard_index==0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

def run():
    nprocs = max(1, os.cpu_count()//2) # number of processors, one process per processor
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        shard = np.empty((shard_size,), dtype=np.uint16) # shard
        target = 0 # index in "shard" where to put block of data

        progress_bar = None

        # pool.imap() is like regular map()
        # fw is list of rows
        # Chunk 1: fw[0]  ... fw[15]
        # Chunk 2: fw[16] ... fw[31]
        # Chunk 3: fw[32] ... fw[47]
        # ...
        # Worker A gets Chunk 1 (16 rows)
        # Worker B gets Chunk 2 (16 rows)
        # "for tokens in ..."" means that tokens0 = tokenize(fw[0]), tokens1 = tokenize(fw[1])...
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            n = len(tokens)
            # put tokens into shard if it fits
            if target + n < shard_size:
                shard[target:target+n] = tokens
                target += n

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                
                progress_bar.update(n)
            else:
                # shard is almost full and won't fit the tokens
                remainder = shard_size - target

                # update and create another line next time
                progress_bar.update(remainder)
                progress_bar = None

                shard[target:target+remainder] = tokens[:remainder]

                write_datafile(shard_index, shard)

                shard_index += 1

                target = n-remainder
                shard[0:target] = tokens[remainder:]

        # if shard is not completely filled - save it anyway
        if target!=0:
            write_datafile(shard_index, shard[:target])

if __name__ == "__main__":
    run()