"""This module contains implementation of ATen ops."""
import torch

# Keys are OpOverload, value is a callable that takes
# XLATensor2
all_ops = {}

# list all Aten ops from pytorch that does mutation
# and need to be implemented in jax

mutation_ops_to_functional = {
    torch.ops.aten.add_: torch.ops.aten.add,
    torch.ops.aten.sub_: torch.ops.aten.sub,
    torch.ops.aten.mul_: torch.ops.aten.mul,
    torch.ops.aten.div_: torch.ops.aten.div,
    torch.ops.aten.pow_: torch.ops.aten.pow,
    torch.ops.aten.lt_: torch.ops.aten.lt,
    torch.ops.aten.le_: torch.ops.aten.le,
    torch.ops.aten.gt_: torch.ops.aten.gt,
    torch.ops.aten.ge_: torch.ops.aten.ge,
    torch.ops.aten.eq_: torch.ops.aten.eq,
    torch.ops.aten.ne_: torch.ops.aten.ne,
}


def make_mutation(op):

  def f(*args, **kwargs):
    res = mutation_ops_to_functional[op](*args, **kwargs)
    args[0].copy_(res)
    return args[0]

  return f


for op in mutation_ops_to_functional.keys():
  all_ops[op] = make_mutation(op)
