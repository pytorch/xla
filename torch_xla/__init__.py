import torch
import _C

torch_zeros_like_builtin = torch.zeros_like


def _torch_xla_zeros_like(p):
    import torch_xla
    if isinstance(p, torch_xla._C.XLATensor):
        return torch_xla._C.XLATensor(torch.zeros(p.size()), device=p.device())
    return torch_zeros_like_builtin(p)


torch.zeros_like = _torch_xla_zeros_like
