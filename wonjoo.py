import torch
import torch_xla

device = "xla:0"
a_xla = torch.tensor([1, 2], device=device)
