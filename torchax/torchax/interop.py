import collections
import copy
import functools
import torch
from inspect import signature
from functools import wraps
from torch.nn.utils import stateless as torch_stateless
import jax
import jax.numpy as jnp
from jax import tree_util as pytree
from jax.experimental.shard_map import shard_map
from torchax import tensor
from torchax import util
from torchax.ops import mappings
import torchax

from torchax.types import JaxValue, TorchValue, JaxCallable, TorchCallable


def extract_all_buffers(m: torch.nn.Module):
  buffers = {}
  params = {}

  def extract_one(module, prefix):
    for k in dir(module):
      try:
        v = getattr(module, k)
      except:
        continue
      qual_name = prefix + k
      if isinstance(v, torch.nn.parameter.Parameter) and v.requires_grad:
        params[qual_name] = v
      elif isinstance(v, torch.Tensor):
        buffers[qual_name] = v
    for name, child in module.named_children():
      extract_one(child, prefix + name + '.')

  extract_one(m, '')
  return params, buffers


def set_all_buffers(m, params, buffers):

  def set_one(module, prefix):
    for k in dir(module):
      qual_name = prefix + k
      if (potential_v := buffers.get(qual_name)) is not None:
        setattr(module, k, potential_v)
      elif (potential_v := params.get(qual_name)) is not None:
        print(k, potential_v)
        setattr(module, k, torch.nn.Parameter(potential_v))
    for name, child in module.named_children():
      set_one(child, prefix + name + '.')

  set_one(m, '')


class JittableModule(torch.nn.Module):

  def __init__(self,
               m: torch.nn.Module,
               extra_jit_args={},
               dedup_parameters=True):
    super().__init__()
    self.params, self.buffers = extract_all_buffers(m)
    self._model = m
    self._jitted = {}

    self._extra_jit_args = extra_jit_args

    self._extra_dumped_weights = {}

    if dedup_parameters:
      temp = collections.defaultdict(list)
      for k, v in self.params.items():
        temp[id(v)].append(k)

      for v in temp.values():
        if len(v) > 1:
          # duplicated weights with different name
          self._extra_dumped_weights[v[0]] = v[1:]
          for extra_keys in v[1:]:
            del self.params[extra_keys]

  @property
  def __class__(self):
    # Lie about the class type so that
    # isinstance(jittable_module, self._model.__class__) works
    return self._model.__class__

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def functional_call(self, method_or_name, params, buffers, *args, **kwargs):
    kwargs = kwargs or {}
    params_copy = copy.copy(params)
    params_copy.update(buffers)
    # reinflate the state dict so there are not any missing keys
    for k, v in self._extra_dumped_weights.items():
      for new_key in v:
        params_copy[new_key] = params_copy[k]

    if isinstance(method_or_name, str):
      method = getattr(self._model, method_or_name)
    else:
      if not callable(method_or_name):
        raise TypeError(
            f"method_or_name should be a callable or a string, got {type(method_or_name)}"
        )
      method = method_or_name
      args = (self._model,) + args
    with torch_stateless._reparametrize_module(self._model, params_copy):
      res = method(*args, **kwargs)
    return res

  def jittable_call(self, method_name: str, *args, **kwargs):
    if method_name not in self._jitted:
      jitted = jax_jit(
          functools.partial(self.functional_call, method_name),
          kwargs_for_jax_jit=self._extra_jit_args,
      )

      def jitted_forward(*args, **kwargs):
        return jitted(self.params, self.buffers, *args, **kwargs)

      self._jitted[method_name] = jitted_forward
    return self._jitted[method_name](*args, **kwargs)

  def forward(self, *args, **kwargs):
    return self.jittable_call('forward', *args, **kwargs)

  def __getattr__(self, key):
    if key == '_model':
      return super().__getattr__(key)
    if key in self._jitted:
      return self._jitted[key]
    return getattr(self._model, key)

  def make_jitted(self, key):
    jitted = jax_jit(
        functools.partial(self.functional_call, key),
        kwargs_for_jax_jit=self._extra_jit_args)

    def call(*args, **kwargs):
      return jitted(self.params, self.buffers, *args, **kwargs)

    self._jitted[key] = call


class CompileMixin:

  def functional_call(self, method, params, buffers, *args, **kwargs):
    kwargs = kwargs or {}
    params_copy = copy.copy(params)
    params_copy.update(buffers)
    with torch_stateless._reparametrize_module(self, params_copy):
      res = method(*args, **kwargs)
    return res

  def jit(self, method):
    jitted = jax_jit(functools.partial(self.functional_call, method_name))

    def call(*args, **kwargs):
      return jitted(self.named_paramters(), self.named_buffers(), *args,
                    **kwargs)

    return call


def compile_nn_module(m: torch.nn.Module, methods=None):
  if methods is None:
    methods = ['forward']

  new_parent = type(
      m.__class__.__name__ + '_with_CompileMixin',
      (CompileMixin, m.__class__),
  )
  m.__class__ = NewParent


def _torch_view(t: JaxValue) -> TorchValue:
  # t is an object from jax land
  # view it as-if it's a torch land object
  if isinstance(t, jax.Array):
    # TODO
    return tensor.Tensor(t, torchax.default_env())
  if isinstance(t, jnp.dtype):
    return mappings.j2t_dtype(t)
  if callable(t):  # t is a JaxCallable
    return functools.partial(call_jax, t)
  # regular types are not changed
  return t


torch_view = functools.partial(pytree.tree_map, _torch_view)


def _jax_view(t: TorchValue) -> JaxValue:
  # t is an object from torch land
  # view it as-if it's a jax land object
  if isinstance(t, torch.Tensor):
    assert isinstance(t, tensor.Tensor) or isinstance(t, tensor.View), type(t)
    return t.jax()
  if isinstance(t, type(torch.int32)):
    return mappings.t2j_dtype(t)

  # torch.nn.Module needs special handling
  if not isinstance(t, torch.nn.Module) and callable(t):  # t is a TorchCallable
    return functools.partial(call_torch, t)
  # regular types are not changed
  return t


jax_view = functools.partial(pytree.tree_map, _jax_view)


def call_jax(jax_func: JaxCallable, *args: TorchValue,
             **kwargs: TorchValue) -> TorchValue:
  args, kwargs = jax_view((args, kwargs))
  res: JaxValue = jax_func(*args, **kwargs)
  return torch_view(res)


def call_torch(torch_func: TorchCallable, *args: JaxValue,
               **kwargs: JaxValue) -> JaxValue:
  args, kwargs = torch_view((args, kwargs))
  with torchax.default_env():
    res: TorchValue = torch_func(*args, **kwargs)
  return jax_view(res)


def j2t_autograd(fn, call_jax=call_jax):
  """Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`.

    It wraps `fn` with `jax.vjp` to compute both the output and residuals (intermediate
    activations). The wrapped function is then run via `call_jax` and integrated into
    the PyTorch autograd framework by saving the residuals into the context object.
    """

  @wraps(fn)
  def inner(*args, **kwargs):
    from jax.tree_util import tree_flatten

    class JaxFun(torch.autograd.Function):

      @staticmethod
      def forward(ctx, tree_def, *flat_args_kwargs):

        tensors, other = util.partition(flat_args_kwargs,
                                        lambda x: isinstance(x, torch.Tensor))
        # We want the arguments that don't require grads to be closured?

        y, fun_vjp = call_jax(_jax_forward, fn, other, tree_def, tensors)

        # Save necessary information for backward
        # Flatten the vjp function. `vjp_spec` contains a jaxpr for the backward pass.
        # `residuals` contains the tensors needed for the backward pass.`
        residuals, vjp_spec = tree_flatten(fun_vjp)
        ctx.vjp_spec = vjp_spec
        ctx.save_for_backward(*residuals)
        return y

      @staticmethod
      def backward(ctx, *grad_out):
        assert len(grad_out) > 0
        grad_out = grad_out if len(grad_out) > 1 else grad_out[0]

        input_grads_structured = call_jax(_jax_backward, ctx.vjp_spec,
                                          ctx.saved_tensors, grad_out)

        # Construct the gradient tuple to be returned.
        # It needs to match the inputs to forward: (tree_def, *flat_inputs)
        # The first gradient (for tree_def) is None.
        # The subsequent gradients correspond to flat_inputs.
        # We need to put a None for inputs that did not require gradients.
        final_grads = [None]
        for needs_grad, grad in zip(
            ctx.needs_input_grad[1:], input_grads_structured, strict=True):
          final_grads.append(grad if needs_grad else None)

        return tuple(final_grads)

    sig = signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat_args_kwargs, tree_def = tree_flatten((bound.args, bound.kwargs))
    y = JaxFun.apply(tree_def, *flat_args_kwargs)
    return y

  return inner


# NOTE(qihqi): This function cannot be inlined from the callsite
#  Becuase if it does, then it won't hit the compilation cache for
#  call_jax. Call jax uses functions' id as key.
def _jax_forward(fn, other, tree_def, tensors):
  """JAX function to compute output and vjp function.

  primals should be a tuple (args, kwargs).
  """
  import jax
  from jax.tree_util import tree_flatten, tree_unflatten

  def fn_wrapper(*tensors):
    # Reconstruct the original args and kwargs
    flat_inputs = util.merge(tensors, other)
    args, kwargs = tree_unflatten(tree_def, flat_inputs)
    return fn(*args, **kwargs)

  return jax.vjp(fn_wrapper, *tensors)


def _jax_backward(vjp_spec, saved_tensors, grad_out):
  """JAX function to compute input gradients.

  Unflattening `saved_tensors` with `vjp_spec` should restore the original vjp function.
  """
  from jax.tree_util import tree_unflatten
  fun_vjp = tree_unflatten(vjp_spec, saved_tensors)
  return fun_vjp(grad_out)


fori_loop = torch_view(jax.lax.fori_loop)


def wrap_jax_jit(torch_function, jax_jit_func=jax.jit, kwargs_for_jax=None):
  kwargs_for_jax = kwargs_for_jax or {}
  jax_func = jax_view(torch_function)
  jitted = jax_jit_func(jax_func, **kwargs_for_jax)
  return torch_view(jitted)


def jax_jit(torch_function,
            kwargs_for_jax_jit=None,
            fix_for_buffer_donation=False):
  return wrap_jax_jit(
      torch_function, jax_jit_func=jax.jit, kwargs_for_jax=kwargs_for_jax_jit)


def jax_shard_map(torch_function, kwargs_for_jax_shard_map=None):
  return wrap_jax_jit(
      torch_function,
      jax_jit_func=shard_map,
      kwargs_for_jax=kwargs_for_jax_shard_map)


def jax_value_and_grad(torch_function, kwargs_for_value_and_grad=None):
  return wrap_jax_jit(
      torch_function,
      jax_jit_func=jax.value_and_grad,
      kwargs_for_jax=kwargs_for_value_and_grad)


def gradient_checkpoint(torch_function, kwargs=None):
  return wrap_jax_jit(
      torch_function, jax_jit_func=jax.checkpoint, kwargs_for_jax=kwargs)
