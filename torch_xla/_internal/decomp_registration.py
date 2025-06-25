import torch
import torch.overrides
from torch_xla.experimental import assume_pure

torch.library.register_kernel("aten::upsample_trilinear3d", "xla",
                              torch._decomp.decompositions.upsample_trilinear3d)


class TorchFunctionOverride(torch.overrides.TorchFunctionMode):

  # Functions to be override with assume_pure
  PURE_OVERRIDE = {
    torch.nn.functional.glu
  }

  def __torch_function__(self, func, types, args, kwargs=None):
    kwargs = kwargs or {}
    if func in TorchFunctionOverride.PURE_OVERRIDE:
      return assume_pure.assume_pure(func)(*args, **kwargs)
    return func(*args, **kwargs)

_mode = TorchFunctionOverride()
_mode.__enter__()