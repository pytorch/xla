import torch

torch.library.register_kernel("aten::upsample_trilinear3d", "XLA",
                              torch._decomp.decompositions.upsample_trilinear3d)
