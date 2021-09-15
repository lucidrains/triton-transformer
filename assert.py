import torch
from triton_transformer import Transformer

assert torch.cuda.is_available()

# instantiate model and data

model = Transformer(
    num_tokens = 256,
    max_seq_len = 1024,
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    causal = True,
    use_triton = False
).cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

# forward and backward pass without triton

logits = model(x)
logits.sum().backward()

logits = logits.clone()
emb_grad = model.token_emb.weight.grad.clone()

model.zero_grad()

# forward and backward pass with triton

triton_logits = model(x, use_triton = True)
triton_logits.sum().backward()

triton_emb_grad = model.token_emb.weight.grad.clone()

# should be equal, for output and gradients on token embeddings

assert torch.allclose(logits.cpu(), triton_logits.cpu()), 'output is the same'
assert torch.allclose(emb_grad.cpu(), triton_emb_grad.cpu()), 'grad is the same'

print('succeeded')
