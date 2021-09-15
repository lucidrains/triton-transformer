import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

import triton
import triton.language as tl

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
