import functools
import torch
import jax
import torch_xla2
from torch_xla2 import interop
from torch_xla2.interop import torch_view, jax_view
import optax


remat = torch_view(jax.remat)
mark_sharding = torch_view(jax.lax.with_sharding_constraint)


def make_train_step(model_fn, 
                    loss_fn, optax_optimizer, 
                    remat_policy=None, 
                    mark_fsdp_sharding_axis=None):
  """Make a function that do one train step given model and loss.

  model_fn: a function representing the model's forward:
      i.e. has signature Callable[weights, buffers, args] -> result. Where,
      weights is a pytree of trainable parameters
      buffers is a pytree of non-trainable parameters / constants
      args is the input data loaded from the data set
      result is the return value of the model
  loss_fn: a function to compute loss.
      i.e. it has signature of Callable[result, label] -> loss
      where, result is what model_fn returned
        loss is loaded from the dataloader.
  optax_optimizer: the optimizer from optax library. for example, optax.adam
  remat_policy: One of jax.ad_checkpoint.checkpoint_policies, specifies how
      to do gradient checkpointing. If None, then it means no checkpointing.
  mark_fsdp_sharding_axis: str. A string name for marking sharding for 
      fsdp. It must be an axis that exists in the current mesh.
      if None, then no sharding is specified (i.e. for single device)
  """
  env = torch_xla2.default_env()
  @functools.partial(
    remat,
    policy=remat_policy)
  def loss(weights, buffers, args, label): # inputs are XLATensor
    with env, jax.named_scope('compute_loss'):
      if mark_fsdp_sharding_axis is not None:
        args = mark_sharding(
            args, 
            jax.sharding.PartitionSpec(mark_fsdp_sharding_axis))
      res = model_fn(weights, buffers, args)
      if mark_fsdp_sharding_axis is not None:
        res = mark_sharding(res, jax.sharding.PartitionSpec(mark_fsdp_sharding_axis))
        label = mark_sharding(label, jax.sharding.PartitionSpec(mark_fsdp_sharding_axis))
      l = loss_fn(res, label)
      return l

  grad_fn = interop.jax_value_and_grad(loss)

  def step(weights, buffers, opt_state, args, label): #inputs are array
    with jax.named_scope('compute_gradient'):
        loss, gradient = grad_fn(weights, buffers, args, label)

    with jax.named_scope("optimizer_updates"):
        updates, opt_state = interop.call_jax(
            optax_optimizer.update,
            gradient, opt_state, weights)
        weights = interop.call_jax(optax.apply_updates, weights, updates)
    return loss, weights, opt_state

  return interop.jax_jit(step, {'donate_argnums': (0, 2)})