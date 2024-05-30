import torch
from torch.utils._pytree import tree_map
import torch_xla
import torch_xla.core.xla_model as xm


def prime_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
  """
  Prime the optimizer state by running a dummy weight update.

  Optimizer state isn't created until after the first training step. Since the
  distributed checkpointing library loads the state_dict in-place, the
  optimizer state must already exist before loading the checkpoint.

  This utility method runs a dummy weight update with zero gradient to ensure
  the optimizer state exists and can be loaded into.

  **Warning** This method calls `optimizer.step`, which can impact the
  optimizer's state and model parameters. Therefore, it should only be used
  prior to restoring a checkpoint, when the state and parameters will be
  immediately overwritten.

  Args:
    optimizer: The optimizer whose state should be primed for checkpoint
               loading.
  """

  # Initial mark_step to ensure all param_groups are backed by device data.
  xm.mark_step()
  xm.wait_device_ops()

  def zero_grad(x):
    if isinstance(x, torch.Tensor) and x.requires_grad:
      x.grad = torch.zeros_like(x, requires_grad=False)
      param_sharding = torch_xla._XLAC._get_xla_op_sharding(x)
      if param_sharding:
        # Match the gradient sharding to the parameter's.
        torch_xla._XLAC._xla_mark_sharding(x.grad, param_sharding)

  tree_map(zero_grad, optimizer.param_groups)
  optimizer.step()
  xm.mark_step()
  xm.wait_device_ops()
  return optimizer
