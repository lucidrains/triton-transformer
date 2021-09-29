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
labels = torch.randint(0, 256, (1, 1024)).cuda()

# forward and backward pass without triton

loss = model(x, labels = labels)
loss.backward()

loss = loss.clone()
emb_grad = model.token_emb.weight.grad.clone()
ln_weight_grad = model.norm.weight.grad.clone()
ln_bias_grad = model.norm.bias.grad.clone()

model.zero_grad()

# forward and backward pass with triton

triton_loss = model(x, labels = labels, use_triton = True)
triton_loss.backward()

triton_emb_grad = model.token_emb.weight.grad.clone()
triton_ln_weight_grad = model.norm.weight.grad.clone()
triton_ln_bias_grad = model.norm.bias.grad.clone()

# should be equal, for output and gradients on token embeddings

assert torch.allclose(loss.cpu(), triton_loss.cpu(), atol=1e-6), 'output is the same'
assert torch.allclose(emb_grad.cpu(), triton_emb_grad.cpu(), atol=2e-6), 'grad is the same'
assert torch.allclose(ln_weight_grad.cpu(), triton_ln_weight_grad.cpu(), atol=2e-6), 'layernorm weight grad is the same'
assert torch.allclose(ln_bias_grad.cpu(), triton_ln_bias_grad.cpu(), atol=2e-6), 'layernorm bias grad is the same'

print('succeeded')
