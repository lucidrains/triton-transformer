import torch
from torch import autograd
import torch.nn.functional as F

from triton_transformer.utils import calc_num_warps, exists

import triton
import triton.language as tl

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

@triton.jit
def relu_squared_activation(x):
    return tl.where(x > 0, x * x, 0.)

class _relu_squared(autograd.Function):
    @classmethod
    def forward(self, ctx, x, w):
        o = triton_bmm(x, w, activation = relu_squared_activation)
        if x.requires_grad:
            ctx.save_for_backward(x, w, o)
        return o

    @classmethod
    def backward(self, ctx, dy):
        x, w, o = ctx.saved_tensors
        dy = torch.sqrt(o) * 2 * dy
        dx = triton_bmm(dy, w.t())
        dw = triton_bmm(x.transpose(-1, -2), dy)
        return dx, dw

triton_relu_squared = _relu_squared.apply

def fused_relu_squared(x, w, use_triton = False):
    if use_triton:
        return triton_relu_squared(x, w)

    return F.relu(x @ w) ** 2
