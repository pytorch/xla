import contextlib
import jax
from jax import dlpack as jaxdl
import jax.numpy as jnp
import numpy
import torch
import torch.func
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree
import torch.utils.dlpack as torchdl

class OperatorNotFound(Exception):
  pass


def wrap(jaxarray):
  return torch_pytree.tree_map_only(jnp.ndarray, XLATensor2, jaxarray)


def unwrap(torchtensors):
  return torch_pytree.tree_map_only(XLATensor2, lambda x: x._elem, torchtensors)


def t2j(t):
  if isinstance(t, XLATensor2):
    return t._elem
  is_bool = False
  if t.dtype == torch.bool:
    is_bool = True
    t = t.to(torch.int8)

  if not t.is_contiguous():
    t = t.contiguous()

  try:
    dl = torchdl.to_dlpack(t)
    res = jaxdl.from_dlpack(dl)
  except Exception:
    # https://github.com/google/jax/issues/7657
    # https://github.com/google/jax/issues/17784
    if t.dtype == torch.bfloat16:
      nparray = (t.cpu().detach().to(torch.float32).numpy()
                )  # numpy don't support bfloat16
    else:
      nparray = t.cpu().detach().numpy()
    res = jnp.asarray(nparray)
    if t.dtype == torch.bfloat16:
      res = res.astype(jnp.bfloat16)

  if is_bool:
    res = res.astype(jnp.bool_)
  return res


def j2t(x):
  try:
    dl = jaxdl.to_dlpack(x)
    res = torchdl.from_dlpack(dl)
  except Exception:
    res = torch.from_numpy(numpy.asarray(x))
  if x.dtype == jnp.bool_:
    res = res.to(torch.bool)
  return res

TORCH_DTYPE_TO_JAX = {
      torch.float16: jnp.dtype('float16'),
      torch.bfloat16: jnp.dtype('bfloat16'),
      torch.half: jnp.dtype('float16'),
      torch.float32: jnp.dtype('float32'),
      torch.double: jnp.dtype('double'),
      torch.long: jnp.dtype('int64'),
      torch.int32: jnp.dtype('int32'),
      torch.int16: jnp.dtype('int16'),
      torch.int8: jnp.dtype('int8'),
      torch.uint8: jnp.dtype('uint8'),
      torch.bool: jnp.dtype('bool_'),
      torch.complex64: jnp.dtype('complex64'),
      torch.complex128: jnp.dtype('complex128'),
      None: None,
}

JAX_DTYPE_TO_TORCH = {value: key for key, value in TORCH_DTYPE_TO_JAX.items()}

def t2j_dtype(dtype):
  if dtype not in TORCH_DTYPE_TO_JAX:
    raise RuntimeError(f'Attempting to convert unknown type: {dtype} to torch type,')
  return TORCH_DTYPE_TO_JAX[dtype]


def j2t_dtype(dtype):
  if dtype not in JAX_DTYPE_TO_TORCH:
    raise RuntimeError(f'Attempting to convert unknown type: {dtype} to torch type,')
  return JAX_DTYPE_TO_TORCH[dtype]


class XLATensor2(torch.Tensor):

  @staticmethod
  def __new__(cls, elem, env):
    dtype = j2t_dtype(elem.dtype)
    shape = list(elem.shape)
    for i, s in enumerate(shape):
      if not isinstance(s, int):
        shape[i] = 1
    if dtype is None:
      dtype = torch.float32
    return torch.Tensor._make_wrapper_subclass(
        cls,
        shape,
        dtype=dtype,
        device='meta',
        requires_grad=False,
    )

  def __init__(self, elem: jax.Array, env: 'Environment'):
    super().__init__()
    self._elem = elem
    self._env = env

  def __str__(self):
    return "XLATensor2({} {})".format(str(type(self._elem)), str(self._elem))

  def __jax_array__(self):
    return self._elem

  @property
  def shape(self):
    return self._elem.shape

  @property
  def ndim(self):
    return len(self._elem.shape)

  def flatten(self, start_dim=0, end_dim=-1):
    if end_dim == -1:
      end_dim = self.ndim
    new_shape = (
        self._elem.shape[:start_dim] + (-1,) + self._elem.shape[end_dim:])
    new_elem = jnp.reshape(self._elem, new_shape)
    return XLATensor2(new_elem, self._env)
    # return torch.reshape(self, new_shape)

  def __setitem__(self, key, val):
    key = unwrap(key)
    self._elem = self._elem.at[key].set(val._elem)

  def type_as(self, other):
    self._elem = self._elem.astype(other._elem.dtype)
    return self

  __torch_function__ = torch._C._disabled_torch_function_impl

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    env = None
    for arg in torch_pytree.arg_tree_leaves(*args, **kwargs):
      if isinstance(arg, XLATensor2):
        env = arg._env
        break

    with env:
      return func(*args, **(kwargs or {}))

  def detach(self):
    return XLATensor2(jax.lax.stop_gradient(self.jax()), self._env)

  def numpy(self) -> numpy.ndarray:
    import numpy as np

    return np.array(self._elem)

  def jax(self) -> jax.Array:
    return self._elem

  def torch(self) -> torch.Tensor:
    return j2t(self.jax())

  def to(self, *args, **kwargs):
    if len(args) == 1:
      if isinstance(args[0], torch.dtype):
        return XLATensor2(self._elem.astype(t2j_dtype(args[0])), self._env)
    if 'dtype' in kwargs:
      dtype = kwargs['dtype']
      return XLATensor2(self._elem.astype(t2j_dtype(dtype)), self._env)
    return self

  @property
  def dtype(self):
    return j2t_dtype(self._elem.dtype)



# TODO: slice of slice should also be another slice
class SliceView(XLATensor2):

  def __init__(self, orig_tensor, slice):
    self._orig_tensor = orig_tensor
    self._slice = slice

  def numpy(self):
    return self._orig_tensor.numpy()[self._slice]

  def jax(self):
    return self._orig_tensor.jax()[self._slice]

  def torch(self):
    return self._orig_tensor.torch()[self.slice]

  def mutate(self, slice, new_content):
    self._orig_tensor._elem = self._orig_tensor.at[slice].set(new_content)


_DEBUG_ACCURACY = False


def debug_accuracy(func, args, kwargs, current_output):
  if not _DEBUG_ACCURACY:
    return True

  args_torch, kwargs_torch, out_torch = torch_pytree.tree_map_only(
      torch.Tensor, lambda x: j2t(x._elem), (args, kwargs, current_output))
  expected_out = func(*args_torch, **kwargs_torch)

  flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
  flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

  for ex, real in zip(flattened_expected_out, flattened_current_out):
    if ex.dtype != real.dtype:
      ex = ex.to(real.dtype)
    try:
      if (_DEBUG_ACCURACY and isinstance(ex, torch.Tensor) and
          not torch.allclose(ex, real, atol=1e-3, equal_nan=True)):
        import pdb

        pdb.set_trace()
    except:
      import pdb

      pdb.set_trace()

  return True


class XLAFunctionMode(torch.overrides.TorchFunctionMode):
  """Context manager that dispatches torch function calls to JAX."""

  def __init__(self, env):
     self.env = env

  def __torch_function__(self,
                         func,
                         types,
                         args=(),
                         kwargs=None) -> torch.Tensor:
    try:
      return self.env.dispatch(func, types, args, kwargs)
    except OperatorNotFound:
      return func(*args, **(kwargs or {}))


class XLADispatchMode(torch_dispatch.TorchDispatchMode):

  def __init__(self, env):
    self.env = env

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if isinstance(func, torch._ops.OpOverloadPacket):
      with self:
        return func(*args, **kwargs)
    if func.namespace != 'aten':
      return func(*args, **kwargs)
    return self.env.dispatch(func, types, args, kwargs)

def _name_of_func(func):
  if hasattr(func, 'name'):
    return func.name()
  return func.__name__


class Environment(contextlib.ContextDecorator):
    """This class holds a set of configurations and "globals" needed

    for executing torch program using jax.
    Things included so far:

    op registry
    PRNGKey
    Configs

    Also helper functions to manipulate those.
    """

    _prng_key: jax.random.PRNGKey


    def __init__(self, random_seed):
        self._prng_key = jax.random.PRNGKey(random_seed)
        self._function_mode = XLAFunctionMode(self)
        self._dispatch_mode = XLADispatchMode(self)

        # name is torch callable
        self._ops = {}
        self.load_ops()

    def load_ops(self):
      from torch_xla2.ops import jaten, jtorch, ops_registry
      self._ops.update(ops_registry.all_aten_ops)
      self._ops.update(ops_registry.all_torch_functions)

      decomps = torch._decomp.core_aten_decompositions() 
      from torch_xla2.decompositions import EXTRA_DECOMP
      decomps.update(EXTRA_DECOMP)
      for k, v in decomps.items():
        if k not in self._ops:
          self._ops[k] = ops_registry.Operator(
            k,
            v,
            is_jax_function=False,
            is_user_defined=False,
            needs_env=False
          )

    def get_and_rotate_prng_key(self):
        self._prng_key, key = jax.random.split(self._prng_key)
        return key

    def dispatch(self, func, types, args, kwargs):
      with jax.named_scope(_name_of_func(func)):
        kwargs = kwargs or {}
        op = self._ops.get(func)

        if op is None and isinstance(func, torch._ops.OpOverloadPacket):
          op = self._ops.get(func.default)

        if op is None and isinstance(func, torch._ops.OpOverload):
          op = self._ops.get(func.overloadpacket)

        if op is None:
          raise OperatorNotFound(
            f'Operator with name {_name_of_func(func)} has no lowering')

        if op.is_jax_function:
          args, kwargs = self.t2j_iso((args, kwargs))

        if op.needs_env:
          kwargs['env'] = self

        with self:
          res = op.func(*args, **kwargs)

        if op.is_jax_function:
          res = self.j2t_iso(res)
        
        #if self.config.debug_accuracy_for_each_op:
        #  debug_accuracy(func, args, kwargs, res)
        return res

    def __enter__(self):
      self._dispatch_mode.__enter__()
      self._function_mode.__enter__()
      return self

    def __exit__(self, *exc):
      self._function_mode.__exit__(*exc)
      self._dispatch_mode.__exit__(*exc)

    def _move_one_value(self, val):
      if isinstance(val, torch.nn.Module):
        state_dict = self.to_xla(val.state_dict())
        val.load_state_dict(state_dict, assign=True)
        return val
      if isinstance(val, XLATensor2):
        return val
      if isinstance(val, torch.Tensor):
        return XLATensor2(t2j(val), self)
      return val

    def to_xla(self, torchvalues):
      # tensors are torch.Tensors (not XLATensor)
      res = torch_pytree.tree_map(
        self._move_one_value,
          torchvalues)
      return res

    def t2j_iso(self, torchtensors):
      return torch_pytree.tree_map_only(
        XLATensor2, lambda x: x.jax(), torchtensors)

    def j2t_iso(self, jaxarray):
      return torch_pytree.tree_map_only(
        jnp.ndarray, lambda x: XLATensor2(x, self), jaxarray)

    def j2t_copy(self, args):
      pass

    def j2t_copy(self, args):
      pass
