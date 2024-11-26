import sys
import contextlib
from typing import Optional, Any
import jax
import jax.numpy as jnp
import numpy
import torch
import torch.distributed._functional_collectives
import torch.func
import torch.utils._mode_utils as mode_utils
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree

from torch_xla2 import config
from torch_xla2.ops import mappings


class OperatorNotFound(Exception):
  pass


def wrap(jaxarray):
  return torch_pytree.tree_map_only(jnp.ndarray, XLATensor2, jaxarray)


def unwrap(torchtensors):
  return torch_pytree.tree_map_only(XLATensor2, lambda x: x._elem, torchtensors)


def t2j(t):
  if isinstance(t, XLATensor2):
    return t._elem
  return mappings.t2j(t)


def j2t(x):
  return mappings.j2t(x)


def t2j_dtype(dtype):
  return mappings.t2j_dtype(dtype)


def j2t_dtype(dtype):
  return mappings.j2t_dtype(dtype)


@contextlib.contextmanager
def log_nested(env, message):
  if env.config.debug_print_each_op:
    print((' ' * log_nested.level) + message, file=sys.stderr)
  log_nested.level += 1
  yield
  log_nested.level -= 1

log_nested.level = 0


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

  __repr__ = __str__

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
    key, val = self._env.t2j_iso((key, val))
    self._elem = self._elem.at[key].set(val)

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

  @property
  def dtype(self):
    return j2t_dtype(self._elem.dtype)

  def dim(self):
    return self.ndim

  @property
  def device(self):
    return torch.device('jax:0')

  @property
  def jax_device(self):
    return self._elem.device

  def tolist(self):
    return self._elem.tolist()
  

 



def debug_accuracy(func, args, kwargs, current_output):
  args_torch, kwargs_torch, out_torch = torch_pytree.tree_map_only(
      torch.Tensor, lambda x: j2t(x._elem), (args, kwargs, current_output))

  with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
    if 'device' in kwargs_torch:
      kwargs_torch['device'] = 'cpu'  # do the torch native for comparison
    expected_out = func(*args_torch, **kwargs_torch)

  flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
  flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

  for ex, real in zip(flattened_expected_out, flattened_current_out):
    if isinstance(ex, torch.Tensor) and ex.dtype != real.dtype:
      ex = ex.to(real.dtype)
    try:
      if (isinstance(ex, torch.Tensor) and
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
    with log_nested(self.env, f'FUNCTION: {_name_of_func(func)}'):
      try:
        return self.env.dispatch(func, types, args, kwargs)
      except OperatorNotFound:
        pass
      if _name_of_func(func) in ('rot90'): # skip rot90 with k%4==0 due to no change
        if len(args) >= 2 and type(args[1]) == int:
          if ((args[1])%4 == 0):
            return args[0]
      return func(*args, **(kwargs or {}))


class XLADispatchMode(torch_dispatch.TorchDispatchMode):

  def __init__(self, env):
    self.env = env

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    with log_nested(self.env, f'DISPATCH: {_name_of_func(func)}'):
      if isinstance(func, torch._ops.OpOverloadPacket):
        with self:
          return func(*args, **kwargs)
      if func.namespace not in ('aten', '_c10d_functional', 'torchvision'):
        return func(*args, **kwargs)
      return self.env.dispatch(func, types, args, kwargs)

def _name_of_func(func):
  if hasattr(func, 'name'):
    return func.name()
  return func.__name__


# Constructors that don't take other tensor as input
TENSOR_CONSTRUCTORS = {
  torch.ones,
  torch.zeros,
  torch.empty,
  torch.empty_strided,
  torch.tensor,
  torch.arange,
  torch.eye,
  torch.randn,
  torch.rand,
  torch.randint,
  torch.full,
  torch.as_tensor,
}


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


    def __init__(self, configuration=None):
        self._function_mode = XLAFunctionMode(self)
        self._dispatch_mode = XLADispatchMode(self)

        # name is torch callable
        self._ops = {}
        self.load_ops()

        self._mesh = None
        self.config = configuration or config.Configuration()

        self._jax_devices = set(['jax', 'jax_cpu', 'xla'])

    def get_as_jax_device(self, device: Any):
      if device is None:
        device = torch.get_default_device()

      if isinstance(device, torch.device):
        device = str(device)
      if (self.config.use_torch_native_for_cpu_tensor and 
          not device.startswith('jax') and not device.startswith('cuda')):
        return None

      if not self.config.treat_cuda_as_jax_device and device.startswith('cuda'):
        return None
      
      if device in ('jax_cpu', 'cpu'):
        return jax.devices('cpu')[0]
      return jax.devices()[0]


    def load_ops(self):
      from torch_xla2.ops import jaten, jtorch, jc10d, jtorchvision_nms, ops_registry
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

    def _to_copy(self, the_tensor, new_dtype, new_device):
      if isinstance(the_tensor, XLATensor2):
        arr = the_tensor.jax()
        if new_dtype is not None and new_dtype != arr.dtype:
          arr = arr.astype(mappings.t2j_dtype(new_dtype))
        if new_device is not None:
          jax_device = self.get_as_jax_device(new_device)
          if jax_device:
            arr = jax.device_put(arr, jax_device)
          else:
            # converting to a non-jax device: let torch native handle it
            torch_tensor = j2t(arr) if isinstance(the_tensor, XLATensor2) else arr
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
              return torch_tensor.to(new_device)
      else:
        if new_dtype is not None and new_dtype != the_tensor.dtype:
          with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
            the_tensor = the_tensor.to(new_dtype)
        jax_device = self.get_as_jax_device(new_device)
        if jax_device:
          arr = t2j(the_tensor)
          arr = jax.device_put(arr, jax_device)
        else:
          with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
            return the_tensor.to(new_device)

      return XLATensor2(arr, self)
      

    def get_and_rotate_prng_key(self, generator: Optional[torch.Generator]=None):
      # Always use the default `randint` to get the next seed
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        next_key = torch.randint(
            0, 2**32, (), dtype=torch.uint32, generator=generator).numpy()

      return jax.random.key(next_key)

    def _handle_tensor_constructor(self, func, args, kwargs):
      device = kwargs.get('device')
      jax_device = self.get_as_jax_device(device)
      if jax_device is None:
        # let torch handle it
        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
          return func(*args, **kwargs)
      with jax.default_device(jax_device):
        op = self._ops.get(func)
        res = op.func(*args, **kwargs)
        if isinstance(res, jax.Array):
          res = XLATensor2(res, self)
        return res

    def _torch_Tensor_to(self, args, kwargs):
      the_tensor = args[0]
      args = args[1:]
      if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        dtype = args[0].dtype
        device = args[0].device
        return self._to_copy(the_tensor, dtype, device)
      device = kwargs.get('device')
      dtype = kwargs.get('dtype')
      # args like pin_memory etc that we will ignore
      args = list(filter(lambda x: not isinstance(x, bool), args))
      if len(args) >= 2:
        device, dtype, *_ = args
      elif len(args) == 1 and isinstance(args[0], torch.dtype):
        dtype = args[0]
      elif len(args) == 1:
        device = args[0]
      return self._to_copy(the_tensor, dtype, device)


    def dispatch(self, func, types, args, kwargs):

      kwargs = kwargs or {}
      if func in TENSOR_CONSTRUCTORS:
        return self._handle_tensor_constructor(func, args, kwargs)
      if func in (torch.Tensor.to, torch.ops.aten.lift_fresh.default ,torch.ops.aten._to_copy, torch.ops.aten._to_copy.default):
        return self._torch_Tensor_to(args, kwargs)

      # If the func doesn't act on XLATensor2, and is not a tensor constructor,
      # We should skip and let torch handle it.
      
      tensor_args = [t for t in torch_pytree.tree_flatten(args)[0] if isinstance(t, torch.Tensor)]
      if tensor_args and all(not isinstance(t, XLATensor2) for t in tensor_args):
        return func(*args, **kwargs)

      with jax.named_scope(_name_of_func(func)):
        op = self._ops.get(func)

        if op is None and isinstance(func, torch._ops.OpOverloadPacket):
          op = self._ops.get(func.default)

        if op is None and isinstance(func, torch._ops.OpOverload):
          op = self._ops.get(func.overloadpacket)

        if op is None:
          raise OperatorNotFound(
            f'Operator with name {_name_of_func(func)} has no lowering')

        old_args, old_kwargs = args, kwargs
        args, kwargs = torch_pytree.tree_map_only(
            torch.distributed._functional_collectives.AsyncCollectiveTensor,
            torch.distributed._functional_collectives.wait_tensor,
            (args, kwargs))
        try:
          if op.is_jax_function:
            args, kwargs = self.t2j_iso((args, kwargs))
        except AssertionError:
          if self.config.debug_mixed_tensor:
            import pdb; pdb.set_trace()
          else:
            raise


        if op.needs_env:
          kwargs['env'] = self

        with self:
          res = op.func(*args, **kwargs)

        if op.is_jax_function:
          res = self.j2t_iso(res)

        if self.config.debug_accuracy_for_each_op:
          debug_accuracy(func, old_args, old_kwargs, res)
        return res

    def __enter__(self):
      self._dispatch_mode.__enter__()
      self._function_mode.__enter__()
      self.enabled = True
      return self

    def __exit__(self, *exc):
      self._function_mode.__exit__(*exc)
      self._dispatch_mode.__exit__(*exc)
      self.enabled = False

    def _move_one_value(self, val):
      if isinstance(val, torch.nn.Module):
        with self:
          return val.to('jax')
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
      def to_jax(x):
        if isinstance(x, torch.distributed._functional_collectives.AsyncCollectiveTensor):
          x = x.wait()
        assert isinstance(x, XLATensor2), f'Expect a XLATensor2 but got {type(x)}; usually this means there is a mixed math between XLATensor and torch.Tensor'
        return x.jax()
      return torch_pytree.tree_map_only(torch.Tensor, to_jax, torchtensors)

    def j2t_iso(self, jaxarray):
      return torch_pytree.tree_map_only(
        jnp.ndarray, lambda x: XLATensor2(x, self), jaxarray)

    def j2t_copy(self, args):
      pass
