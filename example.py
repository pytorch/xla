import torch
import torch_xla

torch.manual_seed(42)

device = 'xla'
x = torch.randn(2, 3, 4, device=device)
x = x + x
print(torch.masked_select(x, x.eq(0)).cpu())
