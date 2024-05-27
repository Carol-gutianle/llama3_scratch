# You can switch __DEBUG__ to False to close all the output in terminal
# 可以将__DEBUG__设置为False关闭打印的信息
__DEBUG__ = True
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional
from types import SimpleNamespace

from tokenizer import Tokenizer

# Here is your parental path of llama-3 model
# 这里是llama-3所在的父目录
base_path = '/mnt/cachenew/gutianle'

# load model here
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load(os.path.join(base_path, 'Meta-Llama-3-8B-Instruct/consolidated.00.pth'))

if __DEBUG__:
    print('Model Keys:')
    print(json.dumps(list(model.keys()), indent=4))
    
# load model params
config = None
with open(os.path.join(base_path, 'Meta-Llama-3-8B-Instruct/params.json'), 'r') as f:
    config = json.load(f)
    
if __DEBUG__:
    print('Model Configs:')
    print(config)

def rms_norm(tensor, norm_weights, norm_eps=config['norm_eps']):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

# 计算RoPE
# rope
def rotary_emb(x: torch.Tensor, rope_theta: int=config['rope_theta']):
    x_pairs = x.float().view(x.shape[0], -1, 2)
    num_tokens, num_parts, _ = x_pairs.shape
    zero_to_one = torch.tensor(range(num_parts)) / num_parts
    freqs = 1.0 / (rope_theta ** zero_to_one)
    freqs_for_each_token = torch.outer(torch.arange(num_tokens), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    
    x_complex_numbers = torch.view_as_complex(x_pairs)
    x_complex_numbers_rotated = x_complex_numbers * freqs_cis
    x_complex_numbers_rotated = torch.view_as_real(x_complex_numbers_rotated)
    return x_complex_numbers_rotated.view(x.shape)

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    norm_eps: float
    rope_theta: float
    
class Attention(nn.Module):
    def __init__(self, args: ModelArgs, i:int) -> None:
        super().__init__()
        self.i = i
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.kv_head_dim = args.dim // self.n_kv_heads
        self.dim = args.dim
        
        self.wq = nn.Linear(
            args.dim, 
            args.n_heads * self.head_dim,
            bias = False
        )
        
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * self.kv_head_dim,
            bias = False
        )
        
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * self.kv_head_dim,
            bias = False
        )
    
    # x: num_tokens, dim
    def forward(self, x):
        qkv_attention_store = []
        # the default of batch_size = 1
        seqlen, _ = x.shape
        # load model weights from model
        xq = model[f'layers.{self.i}.attention.wq.weight']
        xk = model[f'layers.{self.i}.attention.wk.weight']
        xv = model[f'layers.{self.i}.attention.wv.weight']
        # split into multiple heads
        xq_heads = xq.view(self.n_heads, self.head_dim, self.dim)
        xk_heads = xk.view(self.n_kv_heads, xk.shape[0] // self.n_kv_heads, self.dim)
        xv_heads = xv.view(self.n_kv_heads, xv.shape[0] // self.n_kv_heads, self.dim)
            
        for head in range(self.n_heads):
            
            # (head_dim, dim)
            xq_head = xq_heads[head]
            xk_head = xk_heads[head // (self.n_heads // self.n_kv_heads)]
            xv_head = xv_heads[head // (self.n_heads // self.n_kv_heads)]
            
            # q_tokens: (num_tokens, head_dim)
            q_tokens = torch.matmul(x, xq_head.T)
            k_tokens = torch.matmul(x, xk_head.T)
            v_tokens = torch.matmul(x, xv_head.T)
            
            # q_rotated: (num_tokens, head_dim)
            q_rotated = rotary_emb(q_tokens)
            k_rotated = rotary_emb(k_tokens)
            
            scores = torch.matmul(q_rotated, k_rotated.T) / self.head_dim ** 0.5
            
            mask = torch.full((seqlen, seqlen), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            
            scores += mask
            scores = F.softmax(scores.float(), dim=1).to(torch.bfloat16)
            
            output = torch.matmul(scores, v_tokens)
            qkv_attention_store.append(output)
            
        # (num_tokens, dim)
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            
        if __DEBUG__:
            print('The shape of stacked qkv attention is: ', stacked_qkv_attention.shape)
        # (dim, dim)
        wo = model[f'layers.{self.i}.attention.wo.weight']
        # (num_tokens, dim)
        embedding_delta = torch.matmul(stacked_qkv_attention, wo.T)
        return embedding_delta 

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim_multiplier, multiple_of, i) -> None:
        super().__init__()
        self.i = i
        hidden_dim = int(dim * ffn_dim_multiplier)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = None
        self.w2 = None
        self.w3 = None
        
    def forward(self, x):
        self.w1 = model[f'layers.{self.i}.feed_forward.w1.weight']
        self.w2 = model[f'layers.{self.i}.feed_forward.w2.weight']
        self.w3 = model[f'layers.{self.i}.feed_forward.w3.weight']
        final_embedding =  torch.matmul(F.silu(torch.matmul(x, self.w1.T)) * torch.matmul(x, self.w3.T), self.w2.T)
        return final_embedding
    
class TransformerBlock(nn.Module):
    def __init__(self, args:ModelArgs, i:int) -> None:
        super().__init__()
        self.i = i
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, self.i)
        self.feed_forward = FeedForward(self.dim, args.ffn_dim_multiplier, args.multiple_of, self.i)
        self.attention_norm = rms_norm
        self.ffn_norm = rms_norm
        
    def forward(
        self,
        x: torch.Tensor
    ):
        if __DEBUG__:
            print(f'Current working transformer: {self.i}')
        h = x + self.attention(self.attention_norm(x, model[f'layers.{self.i}.attention_norm.weight']))
        out = h + self.feed_forward(self.ffn_norm(h, model[f'layers.{self.i}.ffn_norm.weight']))
        return out
        
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, tokenizer) -> None:
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.dim = params.dim
        self.n_layers = params.n_layers
        
        # embedding layer: transform token list to embedding list
        self.embedding_layer = nn.Embedding(self.vocab_size, self.dim)
        # (num_tokens, embedding_size)
        self.embedding_layer.weight.data.copy_(model['tok_embeddings.weight'])
        self.embedding_layer = self.embedding_layer.to(torch.bfloat16)
        
        # for simplity, there is only one layer here.
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(TransformerBlock(params, i))
        
        self.generate = None
        
        self.tokenizer = tokenizer
        
    def output_layer(self, outputs):
        self.generate = model['output.weight']
        logits = torch.matmul(outputs[-1], self.generate.T)
        next_token = torch.argmax(logits, dim=-1)
        print(f'The next token is {self.tokenizer.decode([next_token])}.')
        return next_token
        
    @torch.inference_mode()
    def forward(self, tokens):
        # (num_tokens, embedding_size)
        h = self.embedding_layer(tokens)
        for layer in self.layers:
            h = layer(h)
        hidden_out = rms_norm(h, model['norm.weight'])
        # decode
        output_token = self.output_layer(hidden_out)
        return output_token
    
if __name__ == '__main__':
    sequence = 'the answer to the ultimate question of life, the universe, and everything is '
    tokenizer = Tokenizer(os.path.join(base_path, 'Meta-Llama-3-8B-Instruct/tokenizer.model'))
    config = SimpleNamespace(**config)
    
    input_ids = [128000] + tokenizer.encode(sequence)
    tokens = torch.tensor(input_ids)
    if __DEBUG__:
        print(f'Input ids are {tokens}')
        print(f'*' * 10)

    llama3 = Transformer(config, tokenizer)
    print(llama3(tokens))
