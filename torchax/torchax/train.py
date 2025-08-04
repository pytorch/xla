import collections
import functools
import torch
import jax
import torchax
from torchax import interop
from torchax.interop import torch_view, jax_view
import optax

remat = torch_view(jax.remat)
mark_sharding = torch_view(jax.lax.with_sharding_constraint)


def make_train_step(model_fn, loss_fn, optax_optimizer, remat_policy=None):
  """Creates a function that performs one training step.

  This function is designed to be used with JAX's `jit` for efficient training.
  It takes a model function, a loss function, and an Optax optimizer, and
  returns a function that computes the loss, calculates gradients, and updates
  the model's weights.

  **Arguments:**

  *   `model_fn`: A function representing the model's forward pass. It should
      have the signature `Callable[weights, buffers, args] -> result`, where:
      *   `weights` is a pytree of trainable parameters.
      *   `buffers` is a pytree of non-trainable parameters and constants.
      *   `args` is the input data from the dataset.
      *   `result` is the model's output.
  *   `loss_fn`: A function to compute the loss. It should have the signature
      `Callable[result, label] -> loss`, where:
      *   `result` is the output of `model_fn`.
      *   `label` is the ground truth from the dataloader.
  *   `optax_optimizer`: An optimizer from the Optax library (e.g., `optax.adam`).
  *   `remat_policy` (optional): A policy from `jax.ad_checkpoint.checkpoint_policies`
      that specifies how to perform gradient checkpointing. If `None`, all
      intermediate activations will be checkpointed.

  **Returns:**

  A function that performs one training step. It has the signature
  `Callable[weights, buffers, opt_state, args, label] -> (loss, new_weights, new_opt_state)`.
  """
  env = torchax.default_env()

  def loss(weights, buffers, args, label):  # inputs are XLATensor
    with env, jax.named_scope('compute_loss'):
      res = model_fn(weights, buffers, args)
      l = loss_fn(res, label)
      return l

  loss = interop.gradient_checkpoint(loss, kwargs={'policy': remat_policy})
  grad_fn = interop.jax_value_and_grad(loss)

  def step(weights, buffers, opt_state, args, label):  #inputs are array
    with jax.named_scope('compute_gradient'):
      loss, gradient = grad_fn(weights, buffers, args, label)

    with jax.named_scope("optimizer_updates"):
      updates, opt_state = interop.call_jax(optax_optimizer.update, gradient,
                                            opt_state, weights)
      weights = interop.call_jax(optax.apply_updates, weights, updates)
    return loss, weights, opt_state

  # TODO: apply jax.jit so the user don't have to.
  return step


class Container:
  pass


class ScannedModule(torch.nn.Module):
  """A `torch.nn.Module` that applies a list of identical modules sequentially.

  This module is designed to be used with `jax.lax.scan` for efficient
  execution of repeated layers. It takes a list of modules, stacks their
  weights, and applies the same module function to the input in a loop.

  **Attributes:**

  *   `checkpoint_policy`: The gradient checkpointing policy to use.
  *   `params`: A `torch.nn.ParameterDict` containing the stacked weights of the
      input modules.
  """

  def __init__(self, module_list, checkpoint_policy=None):
    super().__init__()

    self.c = None
    assert module_list
    self.c = Container()
    self.c.one_mod = module_list[0]
    self.checkpoint_policy = checkpoint_policy

    weights = self._stack_layer_weights(module_list)
    self.layer_weights_keys = list(self.c.one_mod.state_dict().keys())
    self.params = torch.nn.ParameterDict({
        self._param_name_new(k): v for k, v in weights.items()
    })

  def _stack_layer_weights(self, module_list):
    # Create weights such that, for every [n, m] weights
    # becomes [k, n, m] where k is number of layer
    # i.e. stacking layer weights together
    temp = collections.defaultdict(list)
    for m in module_list:
      for k, v in m.state_dict().items():
        temp[k].append(v)
    res = {k: torch.stack(v) for k, v in temp.items()}
    return res

  def _param_name_new(self, old):
    return '___'.join(old.split('.'))

  def _param_name_old(self, new):
    return '.'.join(new.split('___'))

  def forward(self, *args, **kwargs):
    assert not kwargs
    weights = {
        k: self.params[self._param_name_new(k)] for k in self.layer_weights_keys
    }
    scan = interop.torch_view(jax.lax.scan)

    def eval_one_layer(args, weight):
      # unpack args
      h, *rest = args
      newh = torch.func.functional_call(self.c.one_mod, weight, args)
      # next layer's input; and residual to be added to list
      return (newh, *rest), None

    _eval_one_layer = interop.gradient_checkpoint(
        eval_one_layer,
        kwargs={'policy': self.checkpoint_policy},
    )
    h, _ = scan(
        _eval_one_layer,
        args,
        weights,
    )
    return h[0]
