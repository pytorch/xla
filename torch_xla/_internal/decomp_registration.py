import torch

torch.library.register_kernel("aten::upsample_trilinear3d", "xla",
                              torch._decomp.decompositions.upsample_trilinear3d)
