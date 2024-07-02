import aqt.jax.v2.flax.aqt_flax as aqt
import flax
import torch
import torch_xla2


# Copied directly from
# https://github.com/pytorch/xla/blob/77f66b51d9b7571803429139aecb1c6f7e1f375e/torch_xla/distributed/fsdp/utils.py#L125-L159
def apply_xla_patch_to_nn_linear(module, patched_function):
  """
  Recursively apply a patch to the forward pass of `nn.Linear` layers
  to enable using `XLAPatchedLinear.apply` as `torch.nn.functional.linear`,
  so that the backward pass will explicitly use the weight parameter of an
  `nn.Linear` layer to resolve https://github.com/pytorch/xla/issues/3811.

  Without this patch, an `nn.Linear` module in PyTorch/XLA holds and uses
  an intermediate result (rather than the weight parameter) in its backward
  computation, which may break the FSDP's full parameter freeing on it.
  """

  def _try_patching_forward_method(m, forward_method_name="forward"):
    # Check if the module's forward signature is same as in `nn.Linear`
    # (if it has already been modified through other means, we will skip the
    # patch to its forward method here).
    forward_method = getattr(m, forward_method_name, None)
    if forward_method is None:
      return
    if getattr(forward_method, "__func__", None) != torch.nn.Linear.forward:
      return

    from types import MethodType
    patched_forward_method = MethodType(patched_function, m)
    m._nn_linear_forward_original = forward_method
    setattr(m, forward_method_name, patched_forward_method)

  for m in module.modules():  # includes self
    if isinstance(m, torch.nn.Linear):
      _try_patching_forward_method(m, "forward")
      # also handle the case of gradient checkpointing via `checkpoint_module`
      _try_patching_forward_method(m, "_xla_checkpointed_forward_original")

  return module


def quantize_linear_forward(aqt_config, linear, input):
  """
  Run the forward pass of the `torch.nn.Linear` layer using the provided AQT config.
  """
  dot_general = aqt.AqtDotGeneral(aqt_config, prng_name=None)
  params = {'params': {'kernel': linear.weight.T, 'bias': linear.bias}}
  dense_module = flax.linen.Dense(dot_general=dot_general, features=linear.out_features)
  return torch_xla2.interop.call_jax(
      dense_module.apply,
      params,
      input,
  )
