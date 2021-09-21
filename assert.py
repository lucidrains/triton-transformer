import torch
from triton_transformer import Transformer

assert torch.cuda.is_available()

# instantiate model and data

model = Transformer(
    num_tokens = 256,
    max_seq_len = 1024,
    dim = 256,
    depth = 6,
    heads = 8,
    dim_head = 64,
    causal = True,
    use_triton = False
).cuda()

x = torch.randint(0, 256, (1, 512)).cuda()
labels = torch.randint(0, 256, (1, 512)).cuda()

# forward and backward pass without triton

loss = model(x, labels = labels)
loss.backward()

loss = loss.clone()
emb_grad = model.token_emb.weight.grad.clone()

model.zero_grad()

# forward and backward pass with triton

triton_loss = model(x, labels = labels, use_triton = True)
triton_loss.backward()

triton_emb_grad = model.token_emb.weight.grad.clone()

# should be equal, for output and gradients on token embeddings

assert torch.allclose(loss.cpu(), triton_loss.cpu(), atol=1e-6), 'output is the same'
assert torch.allclose(emb_grad.cpu(), triton_emb_grad.cpu(), atol=1e-6), 'grad is the same'

print('succeeded')
