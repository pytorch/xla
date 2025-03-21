import torch_xla

def topp_mask(logits, p, dim=None):
  assert p >= 0 and p <= 1.0, "p must be in [0, 1]."
  if dim is None:
    dim = -1
  return torch_xla._XLAC._xla_topp_mask(logits, p, dim)

