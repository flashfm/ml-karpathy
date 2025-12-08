# First check 01-shakespeare.py since it introduces the Transformer model.

# https://github.com/openai/gpt-2
# Paper: Language Models are Unsupervised Multitask Learners
# Paper: Language Models are Few-Shot Learners
# Paper: Attention is all you need

# GPT-2 was written with Tensor Flow, but Huggingface has its implementation in PyTorch
# https://huggingface.co/openai-community/gpt2
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

from transformers import GPT2LMHeadModel, pipeline, set_seed
from dataclasses import dataclass
from torch.nn import functional as F
import matplotlib.pyplot as plt, torch, torch.nn as nn, tiktoken, math, sys, os, time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import get_file

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

# Let's now reproduce GPT-2 ourselves.
# GPT-2 does not have encoder part and cross-attention part in the decoder block.
# Also, in the GPT-2 paper they tell that thay changed LayerNorm layers positions comparing to original Attention paper.

# Looking at the layer names in the sd_hf dictionary, reproducing the layers:

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
            # But slow on Macbook, had to put under "if"
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

            torch.nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal(module.weight, mean=0.0, std=0.02)

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

# Switch to CUDA if possible
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("Device:", device)

# Let's generate!
# We use Tiktoken tokenizer to get tokens from string and back.

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

def generate(model, x):
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
    x = generate(model, x)
    print_sample(x)

# load_and_generate()
# prints samples that look like in try_huggingface(), though not exactly the same

# Now, let's train an empty model on shakespeare dataset

model = GPT(GPTConfig())
model.to(device)

# adds compilation time, but increases speed on GPU dramatically
# compilation
# a) removes Python interpreter from passes
# b) reduces GPU read/write - decreases memory movements between GPU and GPU's mem (HBM) - also called "kernel fusion" - do more on GPU without moving to HBM
model = torch.compile(model)

text = get_file("shakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# From set of tokens, create batched inputs X and labels Y

def create_training_batch(tokens):
    B,T = 4,32
    buf = torch.tensor(tokens[:B*T + 1])  # +1 so we can create Y as labels
    buf = buf.to(device)
    # View as 2D creates batch rows
    x = buf[:-1].view(B,T)
    y = buf[1:].view(B,T)
    return x,y

# x,y = create_training_batch(enc.encode(text[:1000]))
# logits, loss = model(x, y)
# print(loss)
# since weights are random, should predict any token, so should be roughly ln(1/vocab_size) ~ 11
# prints about 10.8

# Let's optimize

def optimize_single_batch(x, y):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")

# optimize_single_batch(x, y)
# step 49, loss: 0.002874101046472788 - overfitting the single batch (because we have just 1 batch at this point)

class DataLoader:

    def __init__(self, text, B, T):
        self.B = B
        self.T = T

        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        l = len(self.tokens)
        print(f"tokens: {l}")
        print(f"1 epoch = {l // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]  # +1 so we can create Y as labels
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B*T

        # if next batch out of bounds - reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x,y

train_loader = DataLoader(text, B=16, T=1024)

# with the default float32 numbers in matrices GPU does multiplication with highest precision
# we are ok to decrease precision to increase speed
# if so, float32 will be cut to TensorFloat32 (TF32) data type internally in GPU, transparently to our code
# TF32 may not be available on Nvidia GPU prior to A100
# if available, we will have 8x speed up.
# this is only for multiplications in GPU, for other places we use autocast() later below
torch.set_float32_matmul_precision("high")

def gpu_sync():
    if (device == "cuda"):
        torch.cuda.synchronize()
    elif (device == "mps"):
        torch.mps.synchronize()

def optimize():
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # use autocasting of logits to bfloat16 (bf16)
        # important: not float16, otherwise we will need to use "gradient scaler", which makes code more complex
        # weights are still float32, so overall it is called "mixed precision" - some parts are bfloat16, anothers float32
        if device=="cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
                _, loss = model(x, y)

        loss.backward()
        optimizer.step()

        # Python works with GPU with async manner. To calculate time precisely, we want to sync here
        gpu_sync()
        t1 = time.time()
        dt = (t1 - t0) * 1000 # time difference in ms
        tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)

        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_second:.2f}")

optimize()

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
