import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from typing import List, Optional, TypedDict


# "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor",
@torch.library.impl("_c10d_functional::broadcast", "XLA")
def broadcast_xla(self: torch.Tensor,
                  src: int,
                  tag: str,
                  ranks: Optional[List] = None,
                  group_size: Optional[int] = None) -> torch.Tensor:
  assert group_size == None, "currently does not support group_size"
  # xm.collective_broadcast perform an inplace update, but
  # we want an functional implementation here.
  with torch.no_grad():
    scale = torch.tensor(
        1 if xr.global_ordinal() == src else 0, dtype=self.dtype)
    # Transfer scale tensor as device data instead of constant 1 or 0.
    xscale = xm.send_cpu_data_to_device(scale, self.device)[0]
    return xm.all_reduce(xm.REDUCE_SUM, xscale * self, groups=ranks)
