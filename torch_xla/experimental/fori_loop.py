import numpy as np
import torch
import torch_xla
from torch.library import Library, impl

import torch.utils._pytree as pytree

from torch._C import DispatchKey

# from torch._higher_order_ops.utils import (
#     _has_potential_branch_input_alias,
#     _has_potential_branch_input_mutation,
#     _maybe_run_with_interpreter,
#     _set_compilation_env,
#     autograd_not_implemented,
#     UnsupportedAliasMutationException,
# )
from torch._ops import HigherOrderOperator
# from torch._subclasses.fake_tensor import FakeTensorMode
# from torch.fx.experimental.proxy_tensor import (
#     disable_proxy_modes_tracing,
#     make_fx,
#     ProxyTorchDispatchMode,
#     track_tensor_tree,
# )
###

# quantized_decomposed_lib = Library("quantized_decomposed", "IMPL")
while_loop_op = HigherOrderOperator("while_loop")


@while_loop_op.py_impl(DispatchKey.XLA)
def xla_while_loop(cond_fn, body_fn, operands:):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)

def _xla_while_loop(conf_fn, body_fn, operands):
  return torch_xla._XLAC._xla_while_loop_fn(conf_fn, body_fn, operands)

