import torch
from torch import nn, einsum, autograd
import torch.nn.functional as F
from einops import rearrange

import triton
import triton.language as tl

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# triton - fused feedforward (wip)

class _relu_squared(autograd.Function):
    @classmethod
    def forward(self, ctx, x):
        zeros = torch.zeros_like(x)
        out = torch.where(x > 0, x * x, zeros)
        ctx.save_for_backward(x)
        return out

    @classmethod
    def backward(self, ctx, dy):
        x, = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        return torch.where(x > 0, dy * x * 2, zeros)

triton_relu_squared = _relu_squared.apply

# triton - softmax (wip)

@triton.jit
def softmax_kernel_forward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

@triton.jit
def softmax_kernel_backward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    **meta
):
    # todo
    pass

class _softmax(autograd.Function):
    @classmethod
    def forward(self, ctx, x):
        shape = x.shape
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        y = torch.empty_like(x)

        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(y)
        return y.view(*shape)

    @classmethod
    def backward(self, ctx, grad_probs):
        shape = grad_probs.shape
        probs, = ctx.saved_tensors

        dim = grad_probs.shape[-1]
        grad_probs = grad_probs.view(-1, dim)

        w1 = rearrange(probs * grad_probs, 'n d -> () n d ()')
        w2 = torch.eye(dim, dtype = probs.dtype, device = probs.device)[None, ...]
        w2 = w2 - probs[..., None]

        grad = rearrange(w2 @ w1, '() n d () -> n d')
        return grad.view(*shape)

triton_softmax = _softmax.apply

# triton - cross entropy (wip)

def cross_entropy_fn(logits, labels, ignore_index = 0., use_triton = False):
    logits = rearrange(logits, 'b n c -> (b n) c')
    labels = rearrange(labels, 'b n -> (b n)')

    if use_triton:
        loss = triton.ops.cross_entropy(logits, labels)        
    else:
        loss = F.cross_entropy(logits, labels, reduction = 'none')

    mask = (labels != ignore_index)
    return loss.mean()

# triton - layer norm (wip)

class _layernorm(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, eps):
        shape = x.shape
        dim = shape[-1]
        x = x.view(-1, dim)
        x_mean = x.mean(dim = -1, keepdim= True)
        x_var = x.var(dim = -1, unbiased = False, keepdim = True)

        scaled_x = (x - x_mean)
        sqrt_var = (x_var + eps) ** 0.5
        normed_x = scaled_x / sqrt_var
        ctx.save_for_backward(scaled_x, normed_x, gamma, sqrt_var)

        out = rearrange(gamma, 'd -> () d') * normed_x + rearrange(beta, 'd -> () d')
        return out.view(*shape)

    @classmethod
    def backward(cls, ctx, dy):
        shape = dy.shape
        dim = shape[-1]
        dy = dy.view(-1, dim)
        n = dy.shape[0]

        scaled_x, normed_x, gamma, sqrt_var = ctx.saved_tensors

        dbeta = dy.sum(dim = 0)
        dgamma = (dy * normed_x).sum(dim = 0)

        dx = (1 / n) * gamma * (1 / sqrt_var * (n * dy)) - dy.sum(dim = 0) - (scaled_x * ((1 / sqrt_var) ** 2) * (dy * scaled_x).sum(dim = 0))
        dx = dx.view(*shape)
        return dx, dgamma, dbeta, None

def layernorm(x, gamma, beta, eps = 1e-5, use_triton = False):
    if use_triton:
        out = _layernorm.apply(x, gamma, beta, eps)
    else:
        out = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps = eps)
    return out

# helpers classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm_gamma = nn.Parameter(torch.zeros(dim))
        self.norm_beta = nn.Parameter(torch.ones(dim))

        self.act = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads
        x = layernorm(x, self.norm_gamma, self.norm_beta, use_triton = use_triton)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attend_fn = triton_softmax if use_triton else self.act
        attn = attend_fn(sim)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.norm_gamma = nn.Parameter(torch.zeros(dim))
        self.norm_beta = nn.Parameter(torch.ones(dim))

        self.proj_in = nn.Linear(dim, dim * mult)
        self.act = ReLUSquared()
        self.proj_out = nn.Linear(dim * mult, dim)

    def forward(self, x, use_triton = None):
        use_triton = default(use_triton, self.use_triton)

        x = layernorm(x, self.norm_gamma, self.norm_beta, use_triton = use_triton)
        x = self.proj_in(x)

        act_fn = triton_relu_squared if use_triton else self.act
        x = act_fn(x)
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
        use_triton = False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_triton = use_triton),
                FeedForward(dim, use_triton = use_triton)
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

        x = self.norm(x)
        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        loss = cross_entropy_fn(logits, labels, ignore_index = 0, use_triton = use_triton)        
        return loss
