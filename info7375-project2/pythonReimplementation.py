
import os
import sys
import time
import random
import math
import struct
from typing import List
import numpy as np

class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
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


def accum(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a

def silu(x):
    return x * (1 / (1 + np.exp(-x)))
def rmsnorm(out, x, weight):
    norm = np.sqrt(np.mean(np.square(x)) + (1e-5))
    out=weight * (x / norm)
    return out


def softmax(x, size):
    
    max_val = np.max(x)
  
    
    exp_sum = 0.0

    for i in range(size):
        x[i] = math.exp(x[i] - max_val)
        exp_sum += x[i]
    
    for i in range(size):
        x[i] /= exp_sum
    return x


def matmul(xout, x, w, n, d):
    for i in range(d):
        val = 0.0
        for j in range(n):
            val += w[i * n + j] * x[j]
        xout[i] = val
    return xout


class RunState:
    x: List[float]
    xb: List[float]
    q: List[float]
    k: List[float]
    v: List[float]
    att: List[float]
    key_cache: List[float]
    value_cache: List[float]
    xb2: List[float]
    hb: List[float]
    hb2: List[float]
    logits: List[float]



def transformer(token, pos, c, s, w) -> None:
 
    x = s.x
    dim = c.dim
    hidden_dim = c.hidden_dim
    head_size = dim // c.n_heads
 
    kv_dim=(c.dim*c.n_kv_heads)/c.n_heads
    content_row = w.token_embedding_table[token * dim: (token + 1) * dim]
    x[:] = content_row

   


 
    for l in range(c.n_layers):
        loff = l * c.seq_len * dim  
     

        s.xb = rmsnorm(s.xb, x, w.rms_att_weight[l * dim: (l + 1) * dim])

      
        s.q = matmul(s.q, s.xb, w.wq[l * dim * dim: (l + 1) * dim * dim], dim, dim)
        s.k = matmul(s.k, s.xb, w.wk[l * dim * dim: (l + 1) * dim * dim], dim, dim)
        s.v = matmul(s.v, s.xb, w.wv[l * dim * dim: (l + 1) * dim * dim], dim, dim)
   
        for i in range(0, dim, 2):
            head_dim = i % head_size
            freq = 1.0 / (10000 ** (head_dim / head_size))
            angle = pos * freq

            
            head_idx = (i // 2) // head_size
            rotn = 2 if head_idx < c.n_kv_heads else 1

            for v in range(rotn):
                vec = s.q if v == 0 else s.k
                v0, v1 = vec[i], vec[i+1]
                vec[i]   = v0 * math.cos(angle) - v1 * math.sin(angle)
                vec[i+1] = v0 * math.sin(angle) + v1 * math.cos(angle)
     
        s.key_cache[loff + pos * dim: loff + (pos + 1) * dim] = s.k
        s.value_cache[loff + pos * dim: loff + (pos + 1) * dim] = s.v

        for h in range(c.n_heads):
            q = s.q[h * head_size: (h + 1) * head_size]  
            att = s.att[h * c.seq_len: (h + 1) * c.seq_len]
            for t in range(pos + 1):
          
                k = s.key_cache[loff + t * dim + h * head_size: loff + (t + 1) * dim + h * head_size]

                score = 0.0
                for i in range(head_size):
                    score += q[i] * k[i]
                
                score /= math.sqrt(head_size)

          
                att[t] = score

           
            att = softmax(att, pos + 1)

          
           
           
            s.xb[h * head_size : h * head_size + head_size] = 0.0

            for t in range(pos + 1):
            
                v = s.value_cache[loff + t * dim + h * head_size : loff + t * dim + (h+1) * head_size]   
                a = att[t]                         
                for i in range(head_size):
                    s.xb[h * head_size + i] += a * v[i]

     
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


       
 
        for i in range(hidden_dim):
            s.hb[i]=silu(s.hb[i])
        for i in range(hidden_dim):
            s.hb[i]=s.hb[i] * s.hb2[i]

        s.xb = matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim:
                                                         (
                                                                 (l + 1)
                                                                 * dim * hidden_dim
                                                         )], hidden_dim, dim)

      
        x = accum(x, s.xb)

   
    x = rmsnorm(x, x, w.rms_final_weight)

  
    s.logits = matmul(s.logits, x, w.wcls, dim, c.vocab_size)
    return s.logits


def str_lookup(string, vocab):

    try:
        index = vocab.index(string)
        return index
    except ValueError as err:
        return -1


def bpe_encode(text, vocab, vocab_scores):
    tokens = []


    for pos, char in enumerate(text):
        string = char
        id = str_lookup(string, vocab)
        if id == -1:
            print(f"not a good prompt at pos {pos}")
            sys.exit(1)
        tokens.append(id)


    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
        
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            id = str_lookup(string, vocab)
            if id != -1 and vocab_scores[id] > best_score:
           
                best_score = vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break 

 
        tokens[best_idx] = best_id
 
        tokens = tokens[0:best_idx + 1] + tokens[best_idx + 2:]

    return tokens


def time_in_ms():

    return int(time.time() * 1000)


def sample(probabilities):
    n = len(probabilities)
 
    r = random.random()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1  


def argmax(v):
 
    max_i = 0
    max_p = v[0]
    for i in range(1, len(v)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i





def run(args):
    checkpoint = args["checkpoint"]
    temperature = float(args["temperature"])
    steps = int(args["steps"])
    prompt = args["prompt"]

    rng_seed = int(time.time())
    random.seed(rng_seed)

    weights = TransformerWeights()

    with open(checkpoint, "rb") as file:
        
        _config = file.read(struct.calcsize('7i'))
      
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
  
        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)


        shared_weights = 1 if config.vocab_size > 0 else 0
        config.vocab_size = abs(config.vocab_size)

        checkpoint_init_weights(weights, config, file, shared_weights)


    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len


    with open("tokenizer.bin", "rb") as file:
        vocab, vocab_scores = tokenizer_init(config, file)


    state = RunState()
    init_run_state(state, config)

  
    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)

  
    start = 0  
    next_token = 0  
   
    token = 1
    pos = 0  

    

    while pos < steps:
   
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
         
            next_token = prompt_tokens[pos]
        else:
          
            if temperature == 0.0:
  
                next_token = argmax(state.logits)
            else:
                
                state.logits = [i / temperature for i in state.logits]
             
                softmax(state.logits, config.vocab_size)
        
                next_token = sample(state.logits)

        
        token_str = (
            vocab[next_token].lstrip()
            if token == 1 and vocab[next_token][0] == ' ' else vocab[next_token]
        )

        print(token_str, end="")
        sys.stdout.flush()
        
        if next_token == 1:
            break

      
        token = next_token
        pos += 1

       
        if start == 0:
            start = time_in_ms()


    end = time_in_ms()
    print(f"\nachieved tok/s: {(steps - 1) / (end - start) * 1000}")


if __name__ == "__main__":
    args = {
        "checkpoint": './stories15M.bin',
        "temperature": "0.0",
        "steps": "256",
        "prompt": "I love fox"
    }
   
    if len(sys.argv) >= 2:
        args["checkpoint"] = sys.argv[1]

    if len(sys.argv) >= 3:
        args["temperature"] = sys.argv[2]

    if len(sys.argv) >= 4:
        args["steps"] = sys.argv[3]

    if len(sys.argv) >= 5:
        args["prompt"] = sys.argv[4]

    run(args)