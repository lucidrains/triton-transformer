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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask = mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE = BLOCK_SIZE)
    return output

def dropout_fn(x, p, training = False, use_triton = False):
    if p == 0. or not training:
        return x

    if not use_triton:
        return F.dropout(x, p, training = True)

    seed = randrange(int(1e7))
    return seeded_dropout(x, p, seed)
