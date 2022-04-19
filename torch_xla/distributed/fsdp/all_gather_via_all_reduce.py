import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm


def all_gather_via_all_reduce(value, dim=0, groups=None, pin_layout=True):
  """
  This is the old all_gather implementation via all_reduce in PyTorch XLA 1.10 in
  https://github.com/pytorch/xla/blob/v1.10.0/torch_xla/core/xla_model.py#L583-L615,
  which avoids the GRPC error (see https://github.com/pytorch/xla/issues/3423).
  """
  if dim < 0:
    dim = value.dim() + dim
  # use in-place all_reduce on padded_value
  value = value.flatten()
  if not (value.dim() == 1 and dim == 0):
    raise NotImplementedError()

  padded_value = torch.zeros(
      xm.xrt_world_size(),
      value.numel(),
      dtype=value.dtype,
      device=value.device)
  padded_value[xm.get_ordinal()].add_(value)
  xm.all_reduce(
      xm.REDUCE_SUM, [padded_value], groups=groups, pin_layout=pin_layout)
  return padded_value.flatten()
