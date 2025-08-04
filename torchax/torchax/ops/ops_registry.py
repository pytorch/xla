import dataclasses
import logging
from torchax.types import JaxCallable, TorchCallable

from typing import Union, Dict


@dataclasses.dataclass
class Operator:
  """A dataclass that represents a `torchax` operator.

  This class holds the implementation of a PyTorch operator, along with
  metadata that describes how it should be handled by the `torchax` dispatcher.

  **Attributes:**

  *   `torch_op` (`TorchCallable`): The original PyTorch operator.
  *   `func` (`Union[TorchCallable, JaxCallable]`): The implementation of the
      operator, which can be either a PyTorch callable or a JAX callable.
  *   `is_jax_function` (`bool`): `True` if the implementation is a JAX function.
  *   `is_user_defined` (`bool`): `True` if the operator is defined by the user.
  *   `needs_env` (`bool`): `True` if the operator needs access to the `torchax`
      environment.
  *   `is_view_op` (`bool`): `True` if the operator is a view operation.
  """
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
  """Registers a `torch_dispatch` operator.

  This function is used to register an implementation for a PyTorch ATen
  operator.

  **Arguments:**

  *   `aten_op`: The ATen operator to register (e.g., `torch.ops.aten.add`).
  *   `impl_callable`: The implementation of the operator.
  *   `is_jax_function` (`bool`, optional): `True` if the implementation is a JAX
      function. Defaults to `True`.
  *   `is_user_defined` (`bool`, optional): `True` if the operator is defined by
      the user. Defaults to `False`.
  *   `needs_env` (`bool`, optional): `True` if the operator needs access to the
      `torchax` environment. Defaults to `False`.
  *   `is_view_op` (`bool`, optional): `True` if the operator is a view
      operation. Defaults to `False`.

  **Returns:**

  The implementation callable.
  """
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
  """Registers a `torch_function` operator.

  This function is used to register an implementation for a `torch_function`
  operator (e.g., `torch.add`).

  **Arguments:**

  *   `torch_func`: The `torch_function` operator to register.
  *   `impl_callable`: The implementation of the operator.
  *   `is_jax_function` (`bool`, optional): `True` if the implementation is a JAX
      function. Defaults to `True`.
  *   `is_user_defined` (`bool`, optional): `True` if the operator is defined by
      the user. Defaults to `False`.
  *   `needs_env` (`bool`, optional): `True` if the operator needs access to the
      `torchax` environment. Defaults to `False`.
  *   `is_view_op` (`bool`, optional): `True` if the operator is a view
      operation. Defaults to `False`.

  **Returns:**

  The implementation callable.
  """
  op = Operator(
      torch_func,
      impl_callable,
      is_jax_function=is_jax_function,
      is_user_defined=is_user_defined,
      needs_env=needs_env,
      is_view_op=is_view_op)
  all_torch_functions[torch_func] = op
  return impl_callable
