import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from triton_transformer.layernorm import layernorm
from triton_transformer.softmax import softmax
from triton_transformer.cross_entropy import cross_entropy_fn
from triton_transformer.bmm import fused_relu_squared
from triton_transformer.dropout import dropout_fn

from triton_transformer.utils import exists, default

# helpers classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads
        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = use_triton)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attn = softmax(sim, use_triton = use_triton)
        attn = dropout_fn(attn, self.dropout, use_triton = use_triton)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return dropout_fn(out, self.dropout, use_triton = use_triton)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        inner_dim = dim * mult
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.proj_in_weight = nn.Parameter(torch.randn(dim, inner_dim))
        self.proj_in_bias = nn.Parameter(torch.randn(inner_dim))
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(self, x, use_triton = None):
        use_triton = default(use_triton, self.use_triton)

        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = use_triton)

        x = fused_relu_squared(x, self.proj_in_weight, self.proj_in_bias, use_triton = use_triton)
        x = dropout_fn(x, self.dropout, use_triton = use_triton)

        x = self.proj_out(x)
        return x

# main class

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, use_triton = use_triton),
                FeedForward(dim, dropout = ff_dropout, use_triton = use_triton)
            ]))

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        # mask

        self.use_triton = use_triton
        self.causal = causal
        mask = torch.ones(max_seq_len, max_seq_len, dtype = torch.bool).triu(1) if causal else None
        self.register_buffer('mask', mask, persistent = False)

    def forward(
        self,
        x,
        mask = None,
        *,
        labels = None,
        use_triton = None
    ):
        use_triton = default(use_triton, self.use_triton)
        n, device = x.shape[1], x.device

        # embed token and add positional embedding

        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        # generate mask, depending on whether autoregressive or not

        if self.causal:
            mask = self.mask[:n, :n]
            mask = rearrange(mask, 'i j -> () i j')
        elif exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = ~mask

        # go through layers

        for attn, ff in self.layers:
            x = attn(x, mask = mask, use_triton = use_triton) + x
            x = ff(x, use_triton = use_triton) + x

        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = use_triton)
        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        loss = cross_entropy_fn(logits, labels, ignore_index = 0, use_triton = use_triton)
        return loss
