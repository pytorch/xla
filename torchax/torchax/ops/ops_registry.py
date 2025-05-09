import dataclasses
import logging
from torchax.types import JaxCallable, TorchCallable

from typing import Union, Dict


@dataclasses.dataclass
class Operator:
  torch_op: TorchCallable
  func: Union[TorchCallable, JaxCallable]
  is_jax_function: bool
  is_user_defined: bool
  needs_env: bool
  is_view_op: bool


all_aten_ops: Dict[TorchCallable, Operator] = {}
all_torch_functions: Dict[TorchCallable, Operator] = {}


def register_torch_dispatch_op(aten_op,
                               impl_callable,
                               is_jax_function=True,
                               is_user_defined=False,
                               needs_env=False,
                               is_view_op=False):
  op = Operator(
      aten_op,
      impl_callable,
      is_jax_function=is_jax_function,
      is_user_defined=is_user_defined,
      needs_env=needs_env,
      is_view_op=is_view_op)
  if aten_op in all_aten_ops:
    logging.warning(f'Duplicate op registration for {aten_op}')
  all_aten_ops[aten_op] = op
  return impl_callable


def register_torch_function_op(torch_func,
                               impl_callable,
                               is_jax_function=True,
                               is_user_defined=False,
                               needs_env=False,
                               is_view_op=False):
  op = Operator(
      torch_func,
      impl_callable,
      is_jax_function=is_jax_function,
      is_user_defined=is_user_defined,
      needs_env=needs_env,
      is_view_op=is_view_op)
  all_torch_functions[torch_func] = op
  return impl_callable
