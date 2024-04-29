import torch



torch_ops_override = {
    torch.allclose: torch.ops.aten.allclose
}