import numpy as np
import os
import sys
import time
import random
import math
import struct
from typing import List
class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
class TransformerWeights:
    token_embedding_table: List[float]
    rms_att_weight: List[float]
    wq: List[float]
    wk: List[float]
    wv: List[float]
    wo: List[float]
    rms_ffn_weight: List[float]
    w1: List[float]
    w3: List[float]
    w2: List[float]
    rms_final_weight: List[float]
    freq_cis_real: List[float]
    freq_cis_imag: List[float]
    wcls: List[float]



class RunState:
    def __init__(self, config):
        self.x   = [0.0] * config.dim
        self.xb  = [0.0] * config.dim
        self.xb2 = [0.0] * config.dim
        self.hb  = [0.0] * config.hidden_dim
        self.hb2 = [0.0] * config.hidden_dim
        self.q   = [0.0] * config.dim
        self.k   = [0.0] * config.dim
        self.v   = [0.0] * config.dim
        self.att = [0.0] * (config.n_heads * config.seq_len)
        self.logits      = [0.0] * config.vocab_size
        size = config.n_layers * config.seq_len * config.dim
        self.key_cache   = [0.0] * size
        self.value_cache = [0.0] * size

class Transformer:
    def __init__(self, config):
        self.config = config                     
        self.weights = TransformerWeights() 
        self.state = RunState(config)  
def checkpoint_init_weights(weights: TransformerWeights,
                            conf: Config,
                            file,
                            shared_weights: int) -> None:
    def read_floats(count):
        values = struct.unpack(str(count) + 'f', file.read(count * 4 if count > 0 else count))
        return values

    weights.token_embedding_table = read_floats(conf.vocab_size * conf.dim)
    weights.rms_att_weight = read_floats(conf.n_layers * conf.dim)
    weights.wq = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wk = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wv = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.wo = read_floats(conf.n_layers * conf.dim * conf.dim)
    weights.rms_ffn_weight = read_floats(conf.n_layers * conf.dim)
    weights.w1 = read_floats(conf.n_layers * conf.dim * conf.hidden_dim)
    weights.w2 = read_floats(conf.n_layers * conf.hidden_dim * conf.dim)
    weights.w3 = read_floats(conf.n_layers * conf.dim * conf.hidden_dim)
    weights.rms_final_weight = read_floats(conf.dim)
    weights.freq_cis_real = read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2)
    weights.freq_cis_imag = read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2)
    weights.wcls = weights.token_embedding_table if shared_weights else read_floats(-1)
def softmax(x, size):
 
    max_val = x[0]
    for i in range(1, size):
        if x[i] > max_val:
            max_val = x[i]
    # exp and sum
    exp_sum = 0.0
    for i in range(size):
        x[i] = math.exp(x[i] - max_val)
        exp_sum += x[i]
    # normalize
    for i in range(size):
        x[i] /= exp_sum
    return x        
def rmsnorm(x, weight):
    norm = np.sqrt(np.mean(np.square(x)) + (1e-5))
    return weight * (x / norm)

def accum(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
def silu(x):
    return x * (1 / (1 + np.exp(-x)))
def matmul(xout, x, w, n, d):
    for i in range(d):
        val = 0.0
        for j in range(n):
            val += w[i * n + j] * x[j]
        xout[i] = val
    return xout
def argmax(v):
    max_i = 0
    max_p = v[0]
    for i in range(1, len(v)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i
def forward(transformer:Transformer, token, pos):
    p = transformer.config
    w = transformer.weights
    s = transformer.state
    x = s.x
    dim = p.dim
    hidden_dim = p.hidden_dim
    head_size = dim // p.n_heads
    content_row = w.token_embedding_table[token * dim: (token + 1) * dim]
    x[:] = content_row

   
  
    for l in range(p.n_layers):
            s.xb = rmsnorm( x, w.rms_att_weight[l * dim: (l + 1) * dim])
            s.q = matmul(s.q, s.xb, w.wq[l * dim * dim: (l + 1) * dim * dim], dim, dim)
            s.k = matmul(s.k, s.xb, w.wk[l * dim * dim: (l + 1) * dim * dim], dim, dim)
            s.v = matmul(s.v, s.xb, w.wv[l * dim * dim: (l + 1) * dim * dim], dim, dim)

    for i in range(0, dim, 2):
 
        head_dim = i % head_size
  
        freq = 1.0 / (10000 ** (head_dim / head_size))
        angle = pos * freq
       

        rotn = 2 if i < head_size * 2 else 1
        for v in range(rotn):
            vec = s.q if v == 0 else s.k
            v0, v1 = vec[i], vec[i+1]
            vec[i]   = v0 * math.cos(angle) - v1 *  math.sin(angle)
            vec[i+1] = v0 *  math.sin(angle)+ v1 * math.cos(angle)
      
       

    loff = l * p.seq_len * dim  
    s.key_cache[l, pos, :] = s.k
    s.value_cache[l, pos, :] = s.v
    
    for h in range(p.n_heads):
           
            q = s.q[h * head_size: (h + 1) * head_size]

        
            att = s.att[h * p.seq_len: (h + 1) * p.seq_len]

      
            for t in range(pos + 1):
           
                k = s.key_cache[l, t, h * head_size : (h + 1) * head_size]
          
                score = sum(q[i] * k[i] for i in range(head_size))
                score /= math.sqrt(head_size)

              
                att[t] = score

           
            att[h, :pos + 1] = softmax(att[h, :pos + 1])

            xb_ptr = h * head_size
          
            s.xb[xb_ptr: (h + 1) * head_size] = [0.0] * head_size
            for t in range(pos + 1):
             
                v = s.value_cache[l, t, h * head_size : (h + 1) * head_size]
          
                a = att[t]
             
                for i in range(head_size):
                    s.xb[xb_ptr + i] += a * v[i]
         
    s.xb2 = matmul(s.xb2, s.xb, w.wo[l * dim * dim:(l + 1) * dim * dim], dim, dim)

      
    x = accum(x, s.xb2)

      
    s.xb = rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim:(l + 1) * dim])

     
    s.hb = matmul(s.hb, s.xb,
                          w.w1[l * dim * hidden_dim:
                                     (l + 1) * dim * hidden_dim],
                          dim, hidden_dim)

    s.hb2 = matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim:
                                                           (l + 1) * dim * hidden_dim],
                           dim, hidden_dim)

 
    s.hb = silu(s.hb)

      
    s.hb = [s.hb[i] * s.hb2[i] for i in range(hidden_dim)]

    
    s.xb = matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim:
                                                         (
                                                                 (l + 1)
                                                                 * dim * hidden_dim
                                                         )], hidden_dim, dim)

      
    x = accum(x, s.xb)

  
    x = rmsnorm(x, x, w.rms_final_weight)

  
    s.logits = matmul(s.logits, x, w.wcls, dim, p.vocab_size)
def tokenizer_init(conf: Config, file):
    vocab, vocab_scores, max_token_length = [], [], 0

    max_token_length = struct.unpack('i', file.read(4))[0]
    for i in range(0, conf.vocab_size):
        vocab_scores.append(struct.unpack('f', file.read(4))[0])
        len = struct.unpack('i', file.read(4))[0]
        bstr = file.read(len)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        vocab.append(bstr)
    return vocab, vocab_scores, max_token_length
def str_lookup(string, vocab):
    # Find the first perfect match for string in vocab, return its index or -1 if not found
    try:
        index = vocab.index(string)
        return index
    except ValueError as err:
        return -1


def bpe_encode(text, vocab, vocab_scores):
    tokens = []

    # First encode every individual character in the input text
    for pos, char in enumerate(text):
        string = char
        id = str_lookup(string, vocab)
        if id == -1:
            print(f"not a good prompt at pos {pos}")
            sys.exit(1)
        tokens.append(id)

    # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            # string = vocab[tokens[i]].rstrip(b'\x00') + vocab[tokens[i + 1]].rstrip(b'\x00')
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            id = str_lookup(string, vocab)
            if id != -1 and vocab_scores[id] > best_score:
                # This merge pair exists in vocab! Record its score and position
                best_score = vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break  # We couldn't find any more pairs to merge, so we're done

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        tokens = tokens[0:best_idx + 1] + tokens[best_idx + 2:]

    return tokens


def time_in_ms():
    # Returns time in milliseconds for benchmarking the model speed
    return int(time.time() * 1000)


def sample(probabilities):
    n = len(probabilities)
    # Sample index from probabilities, they must sum to 1
    r = random.random()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1  # In case of rounding errors




def init_run_state(state, config):
    state.x = [0.0] * config.dim
    state.xb = [0.0] * config.dim
    state.xb2 = [0.0] * config.dim
    state.hb = [0.0] * config.hidden_dim
    state.hb2 = [0.0] * config.hidden_dim
    state.q = [0.0] * config.dim
    state.k = [0.0] * config.dim
    state.v = [0.0] * config.dim
    state.att = [0.0] * (config.n_heads * config.seq_len)
    state.logits = [0.0] * config.vocab_size
    state.key_cache = [0.0] * (config.n_layers * config.seq_len * config.dim)
    state.value_cache = [0.0] * (config.n_layers * config.seq_len * config.dim)


def run(args):
    checkpoint = args["checkpoint"]
    temperature = float(args["temperature"])
    steps = int(args["steps"])
    prompt = args["prompt"]

    rng_seed = int(time.time())
    random.seed(rng_seed)

    weights = TransformerWeights()
    
    transformer = None
    with open(checkpoint, "rb") as file:
      
        _config = file.read(struct.calcsize('7i'))
   
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
    
        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)

  
        shared_weights = 1 if config.vocab_size > 0 else 0
        config.vocab_size = abs(config.vocab_size)
        transformer = Transformer(config)
        checkpoint_init_weights(transformer.weights, config, file, shared_weights)
     
  
  
    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len


    with open("tokenizer.bin", "rb") as file:
        vocab, vocab_scores, max_token_length = tokenizer_init(config, file)

    transformer.state = RunState(config)
    init_run_state(transformer.state, config)
    state = transformer.state


    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)


    start = 0  
    next_token = 0  

    token = 1
    pos = 0 
  
    print("<s>")

    while pos < steps:
       
        forward(transformer,token, pos)

        if pos < len(prompt_tokens):
            # If we are still processing the input prompt, force the next prompt token
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                next_token = argmax(state.logits)
            else:
                # Apply the temperature to the logits
                state.logits = [i / temperature for i in state.logits]
                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits, config.vocab_size)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

        # Following BOS token (1), sentencepiece decoder strips any leading whitespace
        token_str = (
            vocab[next_token].lstrip()
            if token == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
        )

        print(token_str, end="")
        sys.stdout.flush()
        
        if next_token == 1:
            break

        # Advance forward
        token = next_token
        pos += 1

        # Initialize our timer here because the first iteration could be time consuming due to IO operations
        if start == 0:
            start = time_in_ms()

    # Report achieved tok/s
    end = time_in_ms()
    print(f"\nachieved tok/s: {(steps - 1) / (end - start) * 1000}")


if __name__ == "__main__":
    args = {
        "checkpoint": './stories15M.bin',
        "temperature": "0.0",
        "steps": "256",
        "prompt": "I love fox"
    }
    # if len(sys.argv) < 2:
    #     print(
    #         "Usage: python script.py <checkpoint_file> [temperature] [steps] [prompt]")
    #     sys.exit(1)

    if len(sys.argv) >= 2:
        args["checkpoint"] = sys.argv[1]

    if len(sys.argv) >= 3:
        args["temperature"] = sys.argv[2]

    if len(sys.argv) >= 4:
        args["steps"] = sys.argv[3]

    if len(sys.argv) >= 5:
        args["prompt"] = sys.argv[4]

    run(args)