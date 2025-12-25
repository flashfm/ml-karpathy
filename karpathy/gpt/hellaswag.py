# Using HellaSwag dataset to evaluate our model
# Paper: "HellaSwag: Can a Machine Really Finish Your Sequence" by Rowan Zellers at al.

# Evaluation format

# Question: Context tokens
# Possible answers: option 1, option 2, option 3, option 4

# Our model is small, it cannot choose if we prompt context just once.
# Instead, we give a batch of contexts and see which example has higher probability.

# Context tokens | option 1.......
# Context tokens | option        2 
# Context tokens | option    3....
# Context tokens | option  4......

# B rows with T tokens, if not fit - padded.
# We give this to model, it outputs probabilities for all options,
#   we compare to label (correct option).

import os, json, requests, tiktoken, torch, torch.nn as nn, argparse
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from tqdm import tqdm

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags_format = "https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_{0}.jsonl"
hellaswags = {
    "train": hellaswags_format.format("train"),
    "val": hellaswags_format.format("val"),
    "test": hellaswags_format.format("test")
}

enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, fname: str, chunk_size=1024):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length"), 0)
    with \
        open(fname, "wb") as file, \
        tqdm(desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def render_example(example):
    # tokens - 4xN (4 times context + completion)
    # mask - 1 or 0
    # label - index of correct completion
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    ending_tokens = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # like in GPT-2
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        ending_tokens.append(end_tokens)

    data = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": ending_tokens
    }

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    url = hellaswags[split]
    target = f"hellaswag_{split}.jsonl"
    download_file(url, target)
    with open(os.path.join(DATA_CACHE_DIR, target), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high") # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model)

    num_correct_norm = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)

        logits = model(tokens).logits

        avg_loss, pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)

        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        if num_total<10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")

    # get average loss for completion region (where mask == 1), in each row

    # shift mask, so we start at the last prompt token
    shift_mask = (mask[..., 1:]).contiguous() 
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # now we have loss for each 4 completions, one with lowest loss is most likely
    pred_norm = avg_loss.argmin().item()

    return avg_loss, pred_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
