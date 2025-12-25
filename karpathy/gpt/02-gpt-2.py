# First check 01-shakespeare.py since it introduces the Transformer model.

# https://github.com/openai/gpt-2
# Papers:
# - famous "Attention is all you need"
# - GPT-2 paper: "Language Models are Unsupervised Multitask Learners"
# - GPT-3 paper: "Language Models are Few-Shot Learners"

# GPT-2 was written with Tensor Flow, but Huggingface has its implementation in PyTorch
# https://huggingface.co/openai-community/gpt2
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

from transformers import GPT2LMHeadModel, pipeline, set_seed
from dataclasses import dataclass
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt, torch, torch.nn as nn, tiktoken, math, sys, os, time, inspect
import torch.distributed as dist, numpy as np
from hellaswag import iterate_examples, render_example, get_most_likely_row

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import get_file

# === PART 1 ===
# First let's try HuggingFace implementation and weights of actual GPT-2 and generate some text

def try_huggingface():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # gpt2 is 124 M parameters, gpt2-xl is 1.5 B parameters
    sd_hf = model_hf.state_dict()

    # Prints all layers and weight shapes
    for k,v in sd_hf.items():
        print(k, v.shape)

    # transformer.wte.weight torch.Size([50257, 768]) - means 50257 tokens in vocabulary, 768 is token embedding size
    # transformer.wpe.weight torch.Size([1024, 768]) - means context size is 1024, 768 is position embedding size

    #position_embedding = sd_hf["transformer.wpe.weight"]
    #plt.imshow(position_embedding, cmap="gray") # Positional embedding shows some structure

    # Actual features look like sin/cos charts.

    #plt.plot(position_embedding[:, 150])
    #plt.plot(position_embedding[:, 200])
    #plt.plot(position_embedding[:, 250])
    #plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300], cmap="gray")
    #plt.show()

    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    g = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5, truncation=True)
    print(g)

# try_huggingface()

# === PART 2 ===
# Create model ourselves, but load existing weights.

# GPT-2 does not have encoder part and cross-attention part in the decoder block.
# Also, in the GPT-2 paper they tell that thay changed LayerNorm layers positions comparing to original Attention paper.

# Looking at the layer names in the sd_hf dictionary, reproducing the layers:

# region Layers

@dataclass
class GPTConfig:
    block_size: int = 1024 # context length
    vocab_size: int = 50257 # 50K BPE merges, 256 byte tokens + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections, in all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # we introduce this flag to scale init weights later in code
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask, but in OpenAI/HF naming called "bias"
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        # qkv is for all heads and in batch
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        # nh = number of heads
        # hs = head size
        # C = nh * hs
        # In GPT-2 (124M), n_head = 12, hs = 64, C = 12*64 = 768 channels
        # This is equivalent to MultiHeadAttention from 01-shakespeare, but more effective since (B, nh) is treated as batch dimension and processed in parallel

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        if device!="cuda":
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            # The 4 lines above can be "fused" into a single fused kernel operation called FlashAttention (see below for details)
            # But slow on Macbook, so enabled for CUDA only
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # Paper: Gaussian Error Linear Units (GELUs)
        # The approximate version: https://github.com/pytorch/pytorch/issues/39853
        # It is not really needed now, but was used since the original version was slow in TensorFlow
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # we introduce this flag to scale init weights later in code
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # In the original Attention is all you need paper they refer to another paper:
        # "Using the Output Embedding to Improve Language Models" by Ofir Press and Lior Wolf.
        # The idea is that semantically similar tokens (in the embedding matrix) should get similar probabilities in logits.
        # So the weights of embedding and last linear layer (before softmax) are actually the same matrix.
        self.transformer.wte.weight = self.lm_head.weight # Pointer copy!

        self.apply(self._init_weights)

    # Intitializes weights with random numbers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # normally std = 1/sqrt(number_of_features), number_of_features is embedding matrix size, for 127M model it's 768
            # 1/sqrt(768) ~ 0.036 and 1/sqrt(1600) ~ 0.025
            # but in GPT-2 they used constant 0.02

            std = 0.02

            # why do we multiply by 1/sqrt(N):
            # due to residual stream (x = x + ...) in the Block class code, the standard deviation of values grows over layers
            # example:
            # x = torch.zeros(768)
            # n = 100 # 100 layers
            # for i in range(n):
            #   x += torch.randn(768)
            # print(x.std()) - gives ~ 10
            # but we want ~ 1, so if we do instead x += n ** -0.5 * torch.randn(768)
            # then x.std() will be ~ 1

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5     # we do 2 * ... because  we have x = x + ... two times

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward is needed for generation from the model
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # positions, (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)
        # Classifier (Linear)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:            
            # view(...) transforms (B, T, vocab_size) to (B*T, vocab_size)
            logits_2d = logits.view(-1, logits.size(-1))
            targets_column = targets.view(-1)
            loss = F.cross_entropy(logits_2d, targets_column)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("Loading weights of %s" % model_type)
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),  # 124 M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350 M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280), # 774 M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600), # 1558 M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # ignore the mask

        # load Huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # prepare keys (some weights are transposed)
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys if not k.endswith(".attn.masked_bias")] # ignore the mask
        sd_keys_hf = [k for k in sd_keys if not k.endswith(".attn.bias")] # ignore the mask

        # some weights in TensorFlow are transposed, but we want it back normal
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys)

        # copy weights
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate, device):

        # get parameters that requires gradient
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # all params that has 2D+ shape will be weigth-decay (weights and embdeddings, not biases or layernorms)
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # create AdamW optimizer, preferably fused version
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ("cuda" in device or "mps" in device)
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
#endregion

# region Data Loaders

class ShakespeareDataLoader:

    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        text = get_file("shakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        l = len(self.tokens)
        print(f"tokens: {l}")
        print(f"1 epoch = {l // (B*T)} batches")

        self.init_position = B * T * process_rank
        self.current_position = self.init_position

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]  # +1 so we can create Y as labels
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B * T * self.num_processes

        # if next batch out of bounds - reset
        if self.current_position + ( B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.init_position
        
        return x,y
    
class FineWebEduDataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        assert split in {"train", "val"}
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards)>0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.init_position = B * T * process_rank
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens()
        self.current_position = self.init_position

    def load_tokens(self):
        filename = self.shards[self.current_shard]
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]  # +1 so we can create Y as labels
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B * T * self.num_processes

        # if next batch out of bounds - reset
        if self.current_position + ( B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens()
            self.current_position = self.init_position
        
        return x,y

# endregion

num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding("gpt2")
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def create_prompt(msg):
    tokens = enc.encode(msg) # 8 integers
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)
    return x

def generate_pretrained(model, x):
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            logits = logits[:,-1,:] # last (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # Top-k sampling of 50 (Huggingface default)
            # This means that we sort probs and everything over 50th is replaced to 0, then normalized again
            # This way we have no chance to sample very rare tokens.
            # topk_probs and topk_indices are (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from top-k probabilities
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # ?
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            x = torch.cat((x, xcol), dim=1)
    return x

def print_sample(x):
    B = x.size(0)
    for i in range(B):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

# Let's load weights and generate

def load_and_generate():
    x = create_prompt("Hello, I'm a language model,")
    model = GPT.from_pretrained("gpt2")
    # Switch to evaluation mode.
    # We don't have Batch Norm or others which differ in train/eval mode, but anyway.
    model.eval()
    model.to(device)
    x = generate_pretrained(model, x)
    print_sample(x)

# load_and_generate()
# prints samples that look like in try_huggingface(), though not exactly the same

# === PART 3 ===
# Now, let's train an empty model on Shakespeare dataset, and then on FineWebEdu dataset
# We first do it on 1 GPU, then switch to DDP (multiple GPUs)

# sets up Distributed Data Parallel
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA or MPS is required to use DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    print("device:", device)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# adds compilation time, but increases speed on GPU dramatically
# compilation
# a) removes Python interpreter from passes
# b) reduces GPU read/write - decreases memory movements between GPU and GPU's mem (HBM) - also called "kernel fusion" - do more on GPU without moving to HBM

use_compile = False # temporarily disable compilation (otherwise Hellaswag eval is broken)
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# Let's optimize!
 
# region First, as an example - on single batch

# From set of tokens, create batched inputs X and labels Y
# def create_training_batch(tokens):
#     B,T = 4,32
#     buf = torch.tensor(tokens[:B*T + 1])  # +1 so we can create Y as labels
#     buf = buf.to(device)
#     # View as 2D creates batch rows
#     x = buf[:-1].view(B,T)
#     y = buf[1:].view(B,T)
#     return x,y

# x,y = create_training_batch(enc.encode(text[:1000]))
# logits, loss = model(x, y)
# print(loss)
# since weights are random, should predict any token, so should be roughly ln(1/vocab_size) ~ 11
# prints about 10.8

# def optimize_single_batch(x, y):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#     for i in range(50):
#         optimizer.zero_grad()
#         _, loss = model(x, y)
#         loss.backward()
#         optimizer.step()
#         print(f"step {i}, loss: {loss.item()}")

# optimize_single_batch(x, y)
# step 49, loss: 0.002874101046472788 - overfitting the single batch (because we have just 1 batch at this point)
# endregion

# Then, switch to usual batching approach.

total_batch_size = 524288 # 2**19, ~ 0.5M, in number of tokens

# microbatch size
# increase on big GPUs
B = 16

# context size
T = 1024

# calculate B * T * ddp_world_size (number of GPUs) - if that is less than total_batch_size,
#   then gradient accumulation is not needed
# B = 64 can be used on A100, with no grad accum
# on regular GPU, set to 16

assert total_batch_size % (B * T * ddp_world_size) == 0, "ensure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # see Gradient accumulation below
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

# Use Shakespeare in the beginning, then switch to FineWebEdu
#train_loader = ShakespeareDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
#warmup_steps = 10
#max_steps = 50

train_loader = FineWebEduDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = FineWebEduDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# we're doing 2**19 tokens per step
warmup_steps = 715 # in GPT-3 paper they say they warmup on 375M tokens, 375M/2**19 = 715 
max_steps = 19073 # we have 10e9 total tokens, 10e9/2**19 = 19073

# with the default float32 numbers in matrices GPU does multiplication with highest precision
# we are ok to decrease precision to increase speed
# if so, float32 will be cut to TensorFloat32 (TF32) data type internally in GPU, transparently to our code
# TF32 may not be available on Nvidia GPU prior to A100
# if available, we will have 8x speed up.
# this is only for multiplications in GPU, for other places we use autocast() later below
torch.set_float32_matmul_precision("high")

max_lr = 6e-4
min_lr = max_lr * 0.1

# Setup logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

# Learning rate schedule (taken from GPT-3 paper)
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff should start at 1 and go to 0
    return min_lr + coeff * (max_lr - min_lr)

def gpu_sync():
    if (device == "cuda"):
        torch.cuda.synchronize()
    elif (device == "mps"):
        torch.mps.synchronize()

def optimize():
    # initially we use just AdamW, but then we switch to GPT-3 paper approach with lr decay and other improvements
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    optimizer = raw_model.configure_optimizer(weight_decay = 0.1, learning_rate = 6e-4, device = device)
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 250 == 0 or last_step:
            evaluate(step)

        if (step%100==0 or last_step) and (not use_compile):
            evaluate_hellaswag(step)

        if ((step>0 and step%100==0) or last_step) and (not use_compile):
            generate()

        model.train()
        optimizer.zero_grad()

        # this for-loop added for "gradient accumulation", so we can sequentially process large batches
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # use autocasting of logits to bfloat16 (bf16)
            # important: not float16, otherwise we will need to use "gradient scaler", which makes code more complex
            # weights are still float32, so overall it is called "mixed precision" - some parts are bfloat16, anothers float32
            if device=="cuda":
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                    _, loss = model(x, y)

            # we need to scale loss, because cross-entopy loss has a "mean" (avg) in it, i.e. it multiplies a sum by 1/batch_size
            # we want to compensate for that since have a really large batch, but do small batches

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            # if we are using DDP, we want to let it calculate average grads and communicate it to all processes only at last step
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            loss.backward()

        # since DDP averages gradients, we want to average loss as well to print it later on rank0
        # dist.all_reduce() gets average from all ranks and deposits it back to all ranks (like with gradient)
        if ddp:
            dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)

        # clipping the grad vector norm (aka length) to 1.0 - reason: see below in optimizations
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # our custom learning rate schedule (see below)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        # Python works with GPU with async manner. To calculate time precisely, we want to sync here
        gpu_sync()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = tokens_processed / dt

        if master_process:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_second:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

def evaluate(step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

def evaluate_hellaswag(step):
    # see 04-hellaswag.py
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # process only "our" examples
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(tokens)
                _, pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    acc_norm = num_correct_norm / num_total

    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")

def generate():
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(xgen) # (B, T, vocab_size)
            logits = logits[:,-1,:] # (B, vocab_size) # take logits at last position
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)
    # print
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")

optimize()

if ddp:
    destroy_process_group()

# no compilation, with B=16, T=1024, on Macbook: step 0, loss: 10.935505867004395, dt: 46111.53ms, tok/sec: 355.31

# nvidia-smi tool can show how GPUs are loaded - which GPU in use, how much memory used
# the goal is to saturate memory using batch size (a lot of free mem - increase B, not enough - decrease), use powers of 2 for batch size

# Steps to improve:

# 1. decrease matrix multiplication precision to "high" (TF32 instead of float32)
# on Macbook ~ same stats, since TF32 is not available

# 2. use bfloat16 for calculations
# on Macbook: much slower (dt: 68411.66ms, tok/sec: 239.49) - not clear why, had to disable
# on A100 we expect: dt: 300ms, tok/sec: 54700

# 3. use PyTorch compilation
# on Macbook ~7x speedup: step 0, loss: 10.902034759521484, dt: 6972.98ms, tok/sec: 2349.64

# 4. use FlashAttention
# papers:
# - "Online normalizer calculation for softmax" by Maxim Milakov et al.
# - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" by Tri Dao et al.
# - "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" by Tri Dao
# FlashAttention is kernel fusion operation. It uses more FLOPs but due to reduced use of HBM provides ~7x speedup.
# on Macbook: much slower: dt: 97925.31ms, tok/sec: 167.31 - not clear why, had to disable

# 5. Fix sizes to be powers of 2.
# E.g. default vocab_size was 50257, but we overridden it to nearest divisable to 128 number: 50304
# Expected improvement +4%
# on Macbook: slower

# 6. GPT-3 paper: "Language Models are Few-Shot Learners"
# - Change "betas" argument of AdamW
# - Clip the global norm (aka length of vector) of gradient at 1.0
# - Learning rate schedule (cosine decay with warmup)
# - Skipped: Batch size gradual increase: from 32K tokens to full value over 4-12 B tokens of training. Skipped because it changes number of tokens per step.
# - Add weight decay

# 7. Use gradient accumulation to run any batch size on a small GPU when such large batch size does not fit into memory

# 8. Use multiple GPUs in parallel: DistributedDataParallel (DDP) class, see ddp variable
# Note, when backprop is done, DDP automatically averages gradients calculated on each of the parallel processes
# and then sends it to every process again.
# Since DDP is not batch-aware, we did special optimization to do so only at the end of the gradient accumulation (after grad_accum_steps steps).

# 9. Dataset: Use FineWeb Edu, sample 10BT dataset (10B tokens) instead of Shakespeare (search for FineWeb)
# Also we significantly increase warmup_steps and max_steps since the dataset is much bigger
# We may also increase B if our GPU allows it

# 10. Add validation: validate() and val_data_loader

# 11. Add some sample generation

# 12. Add HellaSwag validation, store stats and validation result into files

# 13. Rewrite everything in C (llm.c) - not covered here