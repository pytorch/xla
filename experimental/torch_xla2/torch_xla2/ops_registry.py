import torch
import torch._decomp as decomp
import torch_xla2.decompositions


class LoweringRegistry:
  def __init__(self):
    self.registered_ops = {}
    self.decomps = {}

  def lookup(self, op_or_name):
    candidate = self._lookup(op_or_name)
    if candidate is None:
      if isinstance(op_or_name, torch._ops.OpOverloadPacket):
        candidate = self._lookup(op_or_name.default)
      if isinstance(op_or_name, torch._ops.OpOverload):
        candidate = self._lookup(op_or_name.overloadpacket)
    return candidate

  def _lookup(self, op):
    candidate = self.registered_ops.get(op)
    if candidate is None:
      candidate = self.decomp.get(op)
    return candidate

  def register(self, op, lowering):
    if isinstance(op, torch._ops.OpOverloadPacket):
      if hasattr(op, "default"):
        self.registered_ops[op.default] = lowering
    self.registered_ops[op] = lowering


lowerings = LoweringRegistry()
EXTRA_DECOMP = decomp.get_decompositions(
  [
    torch.ops.aten.upsample_nearest2d,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._adaptive_avg_pool3d,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.native_dropout,
    torch.ops.aten.reflection_pad1d,
    torch.ops.aten.reflection_pad2d,
    torch.ops.aten.reflection_pad3d,
    torch.ops.aten.replication_pad1d,
    torch.ops.aten.replication_pad2d,
    torch.ops.aten.replication_pad3d,
  ]
)
CORE_ATEN_DECOMP = decomp.core_aten_decompositions()
CORE_ATEN_DECOMP.update(EXTRA_DECOMP)
lowerings.decomp = CORE_ATEN_DECOMP


def _all_core_ops():
  """Yields all core ops."""
  import torch._ops

  for k, v in torch.ops.aten.__dict__.items():
    if k.startswith("__"):
      continue
    if k.startswith("_"):
      continue
    if isinstance(v, torch._ops.OpOverloadPacket):
      for overload in v.overloads():
        op = getattr(v, overload)
        if torch.Tag.core in op.tags:
          yield v
          break


def print_missing_ops():
  core_aten = set(_all_core_ops())
  existing = set(lowerings.registered_ops.keys())
  for v in core_aten - existing:
    print(v)
