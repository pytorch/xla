import torch
from torch._decomp import core_aten_decompositions
from torch._ops import ops, OperatorBase
from typing import Dict, List, Any, Callable, Optional, Tuple

_decompositions: Dict[OperatorBase, Callable] = core_aten_decompositions()


class Wrapper:

  def __init__(self, op: OperatorBase):
    self.op = op
    self.fn = _decompositions[op]

  def __call__(self, args: List[Any], kwargs: Dict[str, Any]):
    try:
      return self.fn(*args, **kwargs)
    except:
      pass


def get_decomposition(operator: str) -> Optional[Wrapper]:
  namespace, rest = operator.split("::")
  op_name, overload = rest.split(".") if "." in rest else (rest, None)

  op = getattr(ops, namespace)
  op = getattr(op, op_name)
  op = getattr(op, overload or "default")

  if op in _decompositions:
    print(f"Found decomposition: {op.name()}")
    return Wrapper(op)

  print(f"Not found decomposition: {op.name()}")


def register_decomposition(op: OperatorBase):

  def wrapper(fn: Callable):
    _decompositions[op] = fn
    print(f"Register decomposition: {op.name()}")
    return fn

  return wrapper


@register_decomposition(ops.aten._embedding_bag.default)
def embedding_bag_default(
    weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Optional[torch.Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  torch._check(indices.dim() in (1, 2), lambda: f"embedding_bag(): indices should be a 1D or 2D tensor.")

  MODE_STR = ["sum", "mean", "amax"]
  embedding_vector_dim = weight.shape[1]

  if indices.dim() == 1:
      torch._check(offsets.dim() == 1, lambda: f"embedding_bag(): offsets should be a 1D tensor.")

      offset2bag = indices.new_zeros(len(indices))
      offset2bag = offset2bag.index_add(0, offsets[1:], offsets.new_ones(len(offsets) - 1))
      offset2bag = offset2bag.cumsum(0)

      bags = len(offsets) - (1 if include_last_offset else 0)

      out = weight.new_empty(bags, embedding_vector_dim)
      out.scatter_reduce_(0, offset2bag, weight, reduce=MODE_STR[mode])
      return out, offset2bag, weight, weight
  else:
      # indices.dim() == 2
      pass
