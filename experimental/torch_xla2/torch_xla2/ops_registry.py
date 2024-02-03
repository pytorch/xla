import torch


class LoweringRegistry:

  def __init__(self):
    self.registered_ops = {}

  def lookup(self, op_or_name):
    candidate = self.registered_ops.get(op_or_name)
    if candidate is None:
      if isinstance(op_or_name, torch._ops.OpOverloadPacket):
        candidate = self.registered_ops.get(op_or_name.default)
      if isinstance(op_or_name, torch._ops.OpOverload):
        candidate = self.registered_ops.get(op_or_name.overloadpacket)
    return candidate

  def register(self, op, lowering):
    self.registered_ops[op] = lowering


lowerings = LoweringRegistry()


def _all_core_ops():
  """Yields all core ops."""
  import torch._ops

  for k, v in torch.ops.aten.__dict__.items():
    if k.startswith('__'):
      continue
    if k.startswith('_'):
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
