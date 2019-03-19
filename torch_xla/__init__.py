import torch
import _XLAC

torch_zeros_like_builtin = torch.zeros_like
torch_clone_builtin = torch.clone


def _torch_xla_zeros_like(p):
    import torch_xla
    if isinstance(p, torch_xla._XLAC.XLATensor):
        return torch_xla._XLAC.XLATensor(torch.zeros(p.size()), device=p.device())
    return torch_zeros_like_builtin(p)

def _torch_xla_clone(p):
    import torch_xla
    if isinstance(p, torch_xla._XLAC.XLATensor):
        return p.clone()
    return torch_clone_builtin(p)


torch.zeros_like = _torch_xla_zeros_like
torch.clone = _torch_xla_clone

_XLAC._initialize_aten_bindings()
