# https://triton-lang.org/getting-started/tutorials/04-low-memory-dropout.html#sphx-glr-getting-started-tutorials-04-low-memory-dropout-py
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from random import randrange

BLOCK_SIZE = 1024

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, **meta):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4

    off0 = block_start + BLOCK_SIZE * 0 + tl.arange(0, BLOCK_SIZE)
    off1 = block_start + BLOCK_SIZE * 1 + tl.arange(0, BLOCK_SIZE)
    off2 = block_start + BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE)
    off3 = block_start + BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE)

    mask0 = off0 < n_elements
    mask1 = off1 < n_elements
    mask2 = off2 < n_elements
    mask3 = off3 < n_elements

    x0 = tl.load(x_ptr + off0, mask = mask0)
    x1 = tl.load(x_ptr + off1, mask = mask1)
    x2 = tl.load(x_ptr + off2, mask = mask2)
    x3 = tl.load(x_ptr + off3, mask = mask3)

    r0, r1, r2, r3 = tl.random.rand4x(seed, off0)
    keep0, keep1, keep2, keep3 = r0 > p, r1 > p, r2 > p, r3 > p

    o0 = tl.where(keep0, x0 / (1 - p), 0.0)
    o1 = tl.where(keep1, x1 / (1 - p), 0.0)
    o2 = tl.where(keep2, x2 / (1 - p), 0.0)
    o3 = tl.where(keep3, x3 / (1 - p), 0.0)

    tl.store(output_ptr + off0, o0, mask = mask0)
    tl.store(output_ptr + off1, o1, mask = mask1)
    tl.store(output_ptr + off2, o2, mask = mask2)
    tl.store(output_ptr + off3, o3, mask = mask3)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 4),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE = BLOCK_SIZE)
    return output

def dropout_fn(x, p, use_triton = False):
    if p == 0. or not x.requires_grad:
        return x

    if not use_triton:
        return F.dropout(x, p, training = True)

    seed = randrange(int(1e6))
    return seeded_dropout(x, p, seed)
