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

# triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    x_ptr, y_ptr, o_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_ol, stride_om, stride_on,
    **meta,
):
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']
    GROUP_SIZE_M = 8

    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak + pid_batch*stride_al)
    y_ptrs = y_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn + pid_batch*stride_bl)

    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)

        x_ptrs += BLOCK_SIZE_K * stride_ak
        y_ptrs += BLOCK_SIZE_K * stride_bk

    if exists(meta['ACTIVATION']):
        o = meta['ACTIVATION'](o)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)

def triton_bmm(x, y, activation = None):
    B, M, K = x.shape

    if y.ndim == 2:
        y = y.unsqueeze(0).expand(B, -1, -1)

    _, K, N = y.shape
    assert (K % 32 == 0), "K must be divisible by 32"

    o = torch.empty((B, M, N), device = x.device, dtype = x.dtype)

    grid = lambda META: (
        B, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    bmm_kernel[grid](
        x, y, o,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        ACTIVATION = activation
    )
    return o

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_with_bias_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    M, N, K,
    stride_xl, stride_xm, stride_xk,
    stride_wl, stride_wk, stride_wn,
    stride_bl, stride_bm, stride_bn,
    stride_ol, stride_om, stride_on,
    **meta,
):
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']
    GROUP_SIZE_M = 8

    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None]*stride_xm + offs_k [None, :]*stride_xk + pid_batch*stride_xl)
    w_ptrs = w_ptr + (offs_k [:, None]*stride_wk + offs_n[None, :]*stride_wn + pid_batch*stride_wl)
    b_ptrs = b_ptr + (offs_m[:, None]*stride_bm + offs_n[None, :]*stride_bn + pid_batch*stride_bl)

    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        o += tl.dot(x, w)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    biases = tl.load(b_ptrs)
    o = o + biases

    if exists(meta['ACTIVATION']):
        o = meta['ACTIVATION'](o)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)

def triton_bmm_with_bias(x, w, b, activation = None):
    B, M, K = x.shape

    if w.ndim == 2:
        w = w.unsqueeze(0).expand(B, -1, -1)

    _, K, N = w.shape

    if b.ndim == 1:
        b = b.unsqueeze(0).unsqueeze(1).expand(B, M, -1)

    assert (K % 32 == 0), "K must be divisible by 32"

    o = torch.empty((B, M, N), device = x.device, dtype = x.dtype)

    grid = lambda META: (
        B, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    bmm_with_bias_kernel[grid](
        x, w, b, o,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        ACTIVATION = activation
    )

    return o

@triton.jit
def relu_squared_activation(x):
    return tl.where(x > 0, x * x, 0.)

class _relu_squared(autograd.Function):
    @classmethod
    def forward(self, ctx, x, w, b):
        o = triton_bmm_with_bias(x, w, b, activation = relu_squared_activation)
        ctx.save_for_backward(x, w, o)
        return o

    @classmethod
    def backward(self, ctx, dy):
        x, w, o = ctx.saved_tensors
        dy = torch.sqrt(o) * 2 * dy
        db = dy.sum(dim = (0, 1))
        dx = triton_bmm(dy, w.t())
        dw = triton_bmm(x.transpose(-1, -2), dy)
        return dx, dw, db

triton_relu_squared = _relu_squared.apply

def fused_relu_squared(x, w, b, use_triton = False):
    if use_triton:
        return triton_relu_squared(x, w, b)

    return F.relu(x @ w + b) ** 2

# triton - softmax

def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps

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
    grad_ptr,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    probs_row = tl.load(input_ptrs, mask = col_offsets < n_cols, other = 0.)
    grad_row = tl.load(grad_ptrs, mask = col_offsets < n_cols, other = 0.)

    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis = 0)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask = col_offsets < n_cols)

class _softmax(autograd.Function):
    @classmethod
    def forward(self, ctx, x):
        shape = x.shape
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        y = torch.empty_like(x)

        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        ctx.save_for_backward(y)
        return y.view(*shape)

    @classmethod
    def backward(self, ctx, grad_probs):
        shape = grad_probs.shape
        probs, = ctx.saved_tensors

        grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(probs)

        softmax_kernel_backward[(n_rows,)](
            dx,
            probs,
            grad_probs,
            grad_probs.stride(0),
            probs.stride(0),
            dx.stride(0),
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE
        )

        return dx.view(*shape)

triton_softmax = _softmax.apply

def softmax(x, use_triton = False):
    if use_triton:
        return triton_softmax(x)
    else:
        return F.softmax(x, dim = -1)

# triton - cross entropy

def cross_entropy_fn(logits, labels, ignore_index = 0., use_triton = False):
    logits = rearrange(logits, 'b n c -> (b n) c')
    labels = rearrange(labels, 'b n -> (b n)')

    if use_triton:
        loss = triton.ops.cross_entropy(logits, labels)        
    else:
        loss = F.cross_entropy(logits, labels, reduction = 'none')

    mask = (labels != ignore_index)
    return loss[mask].mean()

# triton - layer norm

@triton.jit
def layernorm_kernel_forward_training(
    output_ptr,
    mean_centered_ptr,
    inv_var_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    gamma_row_stride,
    beta_row_stride,
    output_row_stride,
    mean_centered_row_stride,
    inv_var_row_stride,
    n_cols,
    eps,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride
    beta_row_start_ptr = beta_ptr + row_idx * beta_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    beta_ptrs = beta_row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.)
    betas = tl.load(beta_ptrs, mask=mask, other=0.)

    row_mean = tl.sum(row, axis = 0) / n_cols
    row_mean_centered = row - row_mean
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis = 0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    output = normed * gammas + betas

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets
    tl.store(mean_centered_ptrs, row_mean_centered, mask=mask)

    inv_var_row_start_ptr = inv_var_ptr + row_idx * inv_var_row_stride
    inv_var_ptrs = inv_var_row_start_ptr + col_offsets
    tl.store(inv_var_ptrs, inv_var, mask=mask)

@triton.jit
def layernorm_kernel_forward_inference(
    output_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    gamma_row_stride,
    beta_row_stride,
    output_row_stride,
    n_cols,
    eps,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride
    beta_row_start_ptr = beta_ptr + row_idx * beta_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    beta_ptrs = beta_row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.)
    betas = tl.load(beta_ptrs, mask=mask, other=0.)

    row_mean = tl.sum(row, axis = 0) / n_cols
    row_mean_centered = row - row_mean
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis = 0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    output = normed * gammas + betas

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def layernorm_kernel_backward(
    output_ptr,
    dy_ptr,
    mean_centered_ptr,
    output_row_stride,
    dy_row_stride,
    mean_centered_row_stride,
    n_cols,
    eps,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    dy_row_start_ptr = dy_ptr + row_idx * dy_row_stride
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = dy_row_start_ptr + col_offsets
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    dy = tl.load(dy_ptrs, mask=mask, other=0.)
    mean_centered = tl.load(mean_centered_ptrs, mask=mask, other=0.)

    row_var = tl.sum(mean_centered * mean_centered, axis = 0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = mean_centered * inv_var

    output = 1. / n_cols * inv_var * (n_cols * dy - tl.sum(dy, axis = 0) - normed * tl.sum(dy * normed, axis = 0))

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

class _layernorm(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, eps, training):
        shape = x.shape
        dim = shape[-1]
        x = x.view(-1, dim)
        n_rows, n_cols = x.shape

        expanded_gamma = gamma[None, :].expand(n_rows, -1)
        expanded_beta = beta[None, :].expand(n_rows, -1)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        out = torch.empty_like(x)

        ctx.training = training
        ctx.eps = eps

        if training:
            scaled_x = torch.empty_like(x)
            inv_var = torch.empty_like(x)

            layernorm_kernel_forward_training[(n_rows,)](
                out,
                scaled_x,
                inv_var,
                x,
                expanded_gamma,
                expanded_beta,
                x.stride(0),
                expanded_gamma.stride(0),
                expanded_beta.stride(0),
                out.stride(0),
                scaled_x.stride(0),
                inv_var.stride(0),
                n_cols,
                eps,
                num_warps = num_warps,
                BLOCK_SIZE = BLOCK_SIZE,
            )
            ctx.save_for_backward(scaled_x, gamma, out)
        else:
            layernorm_kernel_forward_inference[(n_rows,)](
                out,
                x,
                expanded_gamma,
                expanded_beta,
                x.stride(0),
                expanded_gamma.stride(0),
                expanded_beta.stride(0),
                out.stride(0),
                n_cols,
                eps,
                num_warps = num_warps,
                BLOCK_SIZE = BLOCK_SIZE,
            )

        return out.view(*shape)

    @classmethod
    def backward(cls, ctx, dy):
        assert ctx.training, 'forward must be given with training flag of True'

        shape = dy.shape
        dim = shape[-1]
        dy = dy.view(-1, dim)

        scaled_x, gamma, out = ctx.saved_tensors

        dbeta = dy.sum(dim = 0)
        dgamma = (dy * out).sum(dim = 0)
        dxhat = dy * gamma

        n_rows, n_cols = dy.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(dy)

        layernorm_kernel_backward[(n_rows,)](
            dx,
            dxhat,
            scaled_x,
            dx.stride(0),
            dxhat.stride(0),
            scaled_x.stride(0),
            n_cols,
            ctx.eps,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        dx = dx.view(*shape)
        return dx, dgamma, dbeta, None, None

def layernorm(x, gamma, beta, eps = 1e-5, use_triton = False, training = False):
    if use_triton:
        out = _layernorm.apply(x, gamma, beta, eps, training)
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

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads
        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = use_triton, training = self.training)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attn = softmax(sim, use_triton = use_triton)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        inner_dim = dim * mult
        self.norm = nn.LayerNorm(dim)

        self.proj_in_weight = nn.Parameter(torch.randn(dim, inner_dim))
        self.proj_in_bias = nn.Parameter(torch.randn(inner_dim))
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(self, x, use_triton = None):
        use_triton = default(use_triton, self.use_triton)

        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = False, training = self.training)

        x = fused_relu_squared(x, self.proj_in_weight, self.proj_in_bias, use_triton = use_triton)
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

        x = layernorm(x, self.norm.weight, self.norm.bias, use_triton = use_triton, training = self.training)
        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        loss = cross_entropy_fn(logits, labels, ignore_index = 0, use_triton = use_triton)
        return loss
