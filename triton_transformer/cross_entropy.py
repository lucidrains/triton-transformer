import torch
import torch.nn.functional as F
from einops import rearrange

import triton
import triton.language as tl

def cross_entropy_fn(logits, labels, ignore_index = 0., use_triton = False):
    logits = rearrange(logits, 'b n c -> (b n) c')
    labels = rearrange(labels, 'b n -> (b n)')

    if use_triton:
        loss = triton.ops.cross_entropy(logits, labels)        
    else:
        loss = F.cross_entropy(logits, labels, reduction = 'none')

    mask = (labels != ignore_index)
    return loss[mask].mean()
