import torch, torch_xla, torch_xla.core.xla_model as xm
dev = xm.xla_device()
a1 = torch.tensor([[1,0,0,5,0,6]], device=dev)
a2 = torch.nonzero(a1)
print(a2.shape[0])
print("after printing shape")
