import logging
import sys
import contextlib
from typing import Optional, Any
import jax
import jax.numpy as jnp
import numpy
import itertools
import torch
import torch.distributed._functional_collectives
import torch.func
import torch.utils._mode_utils as mode_utils
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree
from torchax.view import View
from torchax import config
from torchax.ops import mappings, ops_registry
from torchax import amp
from jax.experimental import mutable_array

logger = logging.getLogger(__name__)


class OperatorNotFound(Exception):
  pass


def wrap(jaxarray):
  return torch_pytree.tree_map_only(jnp.ndarray, Tensor, jaxarray)


def unwrap(torchtensors):
  return torch_pytree.tree_map_only(Tensor, lambda x: x._elem, torchtensors)


@contextlib.contextmanager
def log_nested(env, message):
  if env.config.debug_print_each_op:
    print((" " * log_nested.level) + message, file=sys.stderr)
  log_nested.level += 1
  yield
  log_nested.level -= 1


log_nested.level = 0


class Tensor(torch.Tensor):

  @staticmethod
  def __new__(cls, elem, env):
    dtype = mappings.j2t_dtype(elem.dtype)
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
        device="meta",
        requires_grad=False,
    )

  def __init__(self, elem: jax.Array, env: "Environment"):
    super().__init__()
    self._elem = elem
    self._env = env

  def __str__(self):
    return "Tensor({} {})".format(str(type(self._elem)), str(self._elem))

  __repr__ = __str__

  def __jax_array__(self):
    return self._elem

  @property
  def shape(self):
    return torch.Size(self._elem.shape)

  @property
  def ndim(self):
    return len(self._elem.shape)

  def flatten(self, start_dim=0, end_dim=-1):
    if end_dim == -1:
      end_dim = self.ndim
    new_shape = (
        self._elem.shape[:start_dim] + (-1,) + self._elem.shape[end_dim + 1:])
    new_elem = jnp.reshape(self._elem, new_shape)
    return Tensor(new_elem, self._env)
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
    # TODO(hanq): figure out why is dispatch mode not sufficient
    if func == torch.ops._c10d_functional.wait_tensor.default:
      return args[0]._env.dispatch(func, types, args, kwargs)
    raise AssertionError(
        'torchax Tensors can only do math within the torchax environment.'
        'Please wrap your code with `with torchax.default_env()` or '
        'call torchax.enable_globally() before.')

  def detach(self):
    return Tensor(jax.lax.stop_gradient(self.jax()), self._env)

  def numpy(self) -> numpy.ndarray:
    import numpy as np

    return np.array(self._elem)

  def jax(self) -> jax.Array:
    return self._elem

  def torch(self) -> torch.Tensor:
    return self._env.j2t_copy(self.jax())

  @property
  def dtype(self):
    return mappings.j2t_dtype(self._elem.dtype)

  def dim(self):
    return self.ndim

  @property
  def device(self):
    return torch.device("jax:0")

  @property
  def jax_device(self):
    return self._elem.device

  @property
  def data(self):
    logger.warning(
        "In-place to .data modifications still results a copy on TPU")
    return self

  @data.setter
  def data(self, other):
    if isinstance(other, Tensor):
      self._elem = other._elem

  def apply_jax(self, jax_function, *args, **kwargs):
    # Call a jax function on _elem
    res = jax_function(self._elem, *args, **kwargs)
    return self._env.j2t_iso(res)

  def apply_jax_(self, jax_function, *args, **kwargs):
    self._elem = jax_function(self._elem, *args, **kwargs)
    return self

  def tolist(self):
    return self._elem.tolist()

  def shard_(self, sharding):
    self.apply_jax_(jax.lax.with_sharding_constraint, sharding)


def debug_accuracy(func, args, kwargs, current_output):
  args_torch, kwargs_torch, out_torch = torch_pytree.tree_map_only(
      torch.Tensor, lambda x: x.torch(), (args, kwargs, current_output))

  with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
    if "device" in kwargs_torch:
      kwargs_torch["device"] = "cpu"  # do the torch native for comparison
    expected_out = func(*args_torch, **kwargs_torch)

  flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
  flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

  for ex, real in zip(flattened_expected_out, flattened_current_out):
    if isinstance(ex, torch.Tensor) and ex.dtype != real.dtype:
      ex = ex.to(real.dtype)
    try:
      if isinstance(ex, torch.Tensor) and not torch.allclose(
          ex, real, atol=1e-3, equal_nan=True):
        import pdb

        pdb.set_trace()
    except:
      import pdb

      pdb.set_trace()

  return True


def _make_debug_msg(is_dispatch, log_args, func, args, kwargs):

  def _display(a):
    if isinstance(a, torch.Tensor):
      return f"Tensor of {type(a)}: {a.dtype}{a.shape}"
    elif isinstance(a, jax.Array):
      return f"Jax Array of {type(a)}: {a.dtype}{a.shape}"
    else:
      return str(a)

  kwargs = kwargs or {}
  title = "DISPATCH" if is_dispatch else "FUNCTION"
  args_msg = "args: " + ",".join(_display(a) for a in args) if log_args else ""
  kwargs_msg = ("kwargs: " +
                ",".join(f"{key}: {_display(a)}" for key, a in kwargs.items())
                if log_args else "")
  return f"{title}: {_name_of_func(func)} {args_msg} ~ {kwargs_msg}"


class XLAFunctionMode(torch.overrides.TorchFunctionMode):
  """Context manager that dispatches torch function calls to JAX."""

  def __init__(self, env):
    self.env = env

  def __torch_function__(self,
                         func,
                         types,
                         args=(),
                         kwargs=None) -> torch.Tensor:
    message = f"FUNCTION: {_name_of_func(func)}"
    if self.env.config.debug_print_each_op_operands:
      message = message + "f"
    message = _make_debug_msg(False,
                              self.env.config.debug_print_each_op_operands,
                              func, args, kwargs)
    with log_nested(self.env, message):
      try:
        return self.env.dispatch(func, types, args, kwargs)
      except OperatorNotFound:
        pass
      if _name_of_func(func) in (
          "rot90"):  # skip rot90 with k%4==0 due to no change
        if len(args) >= 2 and type(args[1]) == int:
          if (args[1]) % 4 == 0:
            return args[0]
      return func(*args, **(kwargs or {}))


class XLADispatchMode(torch_dispatch.TorchDispatchMode):

  def __init__(self, env):
    self.env = env

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    message = _make_debug_msg(True,
                              self.env.config.debug_print_each_op_operands,
                              func, args, kwargs)
    with log_nested(self.env, message):
      if isinstance(func, torch._ops.OpOverloadPacket):
        with self:
          return func(*args, **kwargs)
      # Only functions under these namespaces will be intercepted
      if func.namespace not in (
          "aten",
          "_c10d_functional",
          "torchvision",
          "xla",
      ):
        return func(*args, **kwargs)
      return self.env.dispatch(func, types, args, kwargs)


def _name_of_func(func):
  if hasattr(func, "name"):
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

# TODO(wen): use existing types, either from torch or jax
SUPPORTED_JAX_PLATFROM = ["cpu", "tpu"]


class Environment(contextlib.ContextDecorator):
  """This class holds a set of configurations and "globals" needed

  for executing torch program using jax.
  Things included so far:

  op registry
  PRNGKey
  Configs

  Also helper functions to manipulate those.
  """

  def __init__(self, configuration=None):
    self._function_mode = XLAFunctionMode(self)
    self._dispatch_mode = XLADispatchMode(self)

    # name is torch callable
    self._ops = {}
    self._decomps = {}

    self.load_ops()

    self._mesh = None
    self.config = configuration or config.Configuration()

    self._manually_entered = False
    self.enabled = False

    self._prng_key = mutable_array(
        jax.random.key(torch.initial_seed() % (1 << 63)))
    self.autocast_dtype = None
    self._target_device = jax.local_devices()[0].platform

  @property
  def target_device(self):
    return self._target_device

  @target_device.setter
  def target_device(self, device: str):
    self._target_device = device.lower()

  def manual_seed(self, key):
    self._prng_key = mutable_array(jax.random.key(key))

  @property
  def prng_key(self):
    return self._prng_key[...]

  def get_as_jax_device(self, device: Any):
    if device is None:
      device = torch.get_default_device()

    if isinstance(device, torch.device):
      device = str(device)

    if not self.config.use_torch_native_for_cpu_tensor and device.startswith(
        "cpu"):
      return jax.devices("cpu")[0]

    if self.config.treat_cuda_as_jax_device and device.startswith("cuda"):
      return jax.local_devices()[0]

    if device.startswith("xla"):
      return jax.local_devices()[0]

    # TODO (wen): jax is NOT a device type,
    # once we can register more than one backend, revisit
    if device.startswith("jax"):
      match self.target_device:
        case "cpu":
          return jax.devices("cpu")[0]
        case "tpu":
          return jax.devices("tpu")[0]
        case _:
          raise AttributeError(
              f"Cannot handle env.target_device {self.target_device}")

    return None  # fallback to torch

  def load_ops(self):
    from torchax.ops import jaten, jtorch, jc10d, jtorchvision_nms

    for k, v in itertools.chain(ops_registry.all_aten_ops.items(),
                                ops_registry.all_torch_functions.items()):
      if v.is_jax_function:
        self._ops[k] = v
      else:
        self._decomps[k] = v

    from torchax.decompositions import DECOMPOSITIONS, MUTABLE_DECOMPOSITION

    for k, v in DECOMPOSITIONS.items():
      if k not in self._decomps:
        self._decomps[k] = ops_registry.Operator(
            k,
            v,
            is_jax_function=False,
            is_user_defined=False,
            needs_env=False,
            is_view_op=k in MUTABLE_DECOMPOSITION,
        )

  def _get_op_or_decomp(self, func):

    def _get_from_dict(op_dict, op):
      op = op_dict.get(func)
      if op is None and isinstance(func, torch._ops.OpOverloadPacket):
        op = op_dict.get(func.default)
      if op is None and isinstance(func, torch._ops.OpOverload):
        op = op_dict.get(func.overloadpacket)
      return op

    op = _get_from_dict(self._ops, func)

    if op is None:
      # fallback to decompose
      op = _get_from_dict(self._decomps, func)

    if op is None:
      raise OperatorNotFound(
          f"Operator with name {_name_of_func(func)} has no lowering")

    return op

  def _to_copy(self, the_tensor, new_dtype, new_device):
    if isinstance(the_tensor, View):
      the_tensor = the_tensor.torch()

    if isinstance(the_tensor, Tensor):

      arr = the_tensor.jax()

      if new_dtype is not None and new_dtype != arr.dtype:
        arr = arr.astype(mappings.t2j_dtype(new_dtype))

      if new_device is not None:
        match str(new_device).lower():
          case "cpu":
            # converting to a non-jax device: let torch native handle it
            torch_tensor = self.j2t_copy(arr) if isinstance(the_tensor,
                                                            Tensor) else arr
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
              return torch_tensor.to(new_device)
          case "jax":
            # move torchax.tensor / jax tensor between devices
            # I don't know ifgit  this will work after the model is jitted
            if self.target_device != the_tensor.jax_device.platform:
              arr = jax.device_put(the_tensor.jax(),
                                   jax.devices(self.target_device)[0])
              return Tensor(arr, self)
          case _:
            logging.error(f"torchax.Tenosr cannot handle device {new_device}")

    else:
      if new_dtype is not None and new_dtype != the_tensor.dtype:
        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
          the_tensor = the_tensor.to(new_dtype)

      if new_device is None:  ## device is None means don't change device
        return the_tensor

      jax_device = self.get_as_jax_device(new_device)
      if jax_device:
        arr = self.t2j_copy(the_tensor)
        arr = jax.device_put(arr, jax_device)
      else:
        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
          return the_tensor.to(new_device)

    return Tensor(arr, self)

  def get_and_rotate_prng_key(self,
                              generator: Optional[torch.Generator] = None):
    if generator is not None:
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        self._prng_key[...] = jax.random.key(generator.initial_seed() % (2**63))
    old_key = self._prng_key[...]
    new_prng_key, next_key = jax.random.split(old_key)
    self._prng_key[...] = new_prng_key
    return next_key

  def _handle_tensor_constructor(self, func, args, kwargs):
    device = kwargs.get("device")
    jax_device = self.get_as_jax_device(device)
    # TODO(qihqi) figure out better ways for device propagation
    if not self._manually_entered and jax_device is None:
      # let torch handle it
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        return func(*args, **kwargs)
    with jax.default_device(jax_device):
      requires_grad = kwargs.get("requires_grad", False)
      op = self._get_op_or_decomp(func)
      res = op.func(*args, **kwargs)
      if isinstance(res, jax.Array):
        res = Tensor(res, self)
      if requires_grad:
        res.requires_grad = True
      return res

  def _torch_Tensor_to(self, args, kwargs):
    the_tensor = args[0]
    args = args[1:]
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
      dtype = args[0].dtype
      device = args[0].device
      return self._to_copy(the_tensor, dtype, device)
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
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
    if func in (
        torch.Tensor.to,
        torch.ops.aten.lift_fresh.default,
        torch.ops.aten._to_copy,
        torch.ops.aten._to_copy.default,
    ):
      return self._torch_Tensor_to(args, kwargs)

    # If the func doesn't act on Tensor, and is not a tensor constructor,
    # We should skip and let torch handle it.

    tensor_args = [
        t for t in torch_pytree.tree_flatten(args)[0]
        if isinstance(t, torch.Tensor)
    ]

    def is_not_torchax_tensor(x):
      return not isinstance(x, Tensor) and not isinstance(x, View)

    if tensor_args and all(is_not_torchax_tensor(t) for t in tensor_args):
      res = func(*args, **kwargs)
      return res

    with jax.named_scope(_name_of_func(func)):
      op = self._get_op_or_decomp(func)

      old_args, old_kwargs = args, kwargs
      with self._dispatch_mode:
        args, kwargs = torch_pytree.tree_map_only(
            torch.distributed._functional_collectives.AsyncCollectiveTensor,
            torch.distributed._functional_collectives.wait_tensor,
            (args, kwargs),
        )

      try:
        if not op.is_view_op:
          args, kwargs = self.v2t_iso((args, kwargs))

        with self:
          if self.autocast_dtype is not None:
            autocast_policy = amp.autocast_policy.get(func)
            if autocast_policy is not None:
              args, kwargs = amp.execute_policy(autocast_policy, args, kwargs,
                                                self.autocast_dtype)

        if op.is_jax_function:
          args, kwargs = self.t2j_iso((args, kwargs))
      except AssertionError:
        if self.config.debug_mixed_tensor:
          breakpoint()
        else:
          raise

      if op.needs_env:
        kwargs["env"] = self

      if op.is_jax_function:
        res = op.func(*args, **kwargs)
      else:
        # enable dispatch mode because this op could be a composite autograd op
        # meaning, it will decompose in C++
        with self._dispatch_mode:
          res = op.func(*args, **kwargs)

      if op.is_jax_function:
        res = self.j2t_iso(res)

      if self.config.force_materialize_views and isinstance(res, View):
        res = res.torch()

      if self.config.debug_accuracy_for_each_op:
        debug_accuracy(func, old_args, old_kwargs, res)
      return res

  def enable_torch_modes(self):
    self._dispatch_mode.__enter__()
    self._function_mode.__enter__()
    self.enabled = True

  def disable_torch_modes(self, *exc):
    if not exc:
      exc = (None, None, None)
    self._function_mode.__exit__(*exc)
    self._dispatch_mode.__exit__(*exc)
    self.enabled = False

  def __enter__(self):
    self.enable_torch_modes()
    self._manually_entered = True
    return self

  def __exit__(self, *exc):
    self._manually_entered = False
    self.disable_torch_modes(*exc)

  def _move_one_value(self, val):
    if isinstance(val, torch.nn.Module):
      with self:
        return val.to("jax")
    if isinstance(val, Tensor):
      return val
    if isinstance(val, torch.Tensor):
      return Tensor(self.t2j_copy(val), self)
    return val

  def to_xla(self, torchvalues):
    # tensors are torch.Tensors (not XLATensor)
    res = torch_pytree.tree_map(self._move_one_value, torchvalues)
    return res

  def t2j_iso(self, torchtensors):
    """Convert torchax Tensor to jax array.
    
    This function will not copy, will just unwrap the inner jax array out.
    Note: iso is short for "isomorphic"
    """

    def to_jax(x):
      if isinstance(
          x, torch.distributed._functional_collectives.AsyncCollectiveTensor):
        x = x.wait()
      assert isinstance(x, Tensor) or isinstance(x, View), (
          f"Expect a Tensor or a View but got {type(x)}; usually this means there is a mixed math between XLATensor and torch.Tensor"
      )
      return x.jax()

    res = torch_pytree.tree_map_only(torch.Tensor, to_jax, torchtensors)
    return res

  def v2t_iso(self, views):

    def to_tensor(x):
      if isinstance(x, View):
        return x.torch()
      return x

    res = torch_pytree.tree_map_only(View, to_tensor, views)
    return res

  def j2t_iso(self, jaxarray):
    """Convert jax array to torchax Tensor.
    
    This function will not copy, will just wrap the jax array with a torchax Tensor
    Note: iso is short for "isomorphic"
    """
    return torch_pytree.tree_map_only(jax.Array, lambda x: Tensor(x, self),
                                      jaxarray)

  def j2t_copy(self, args):
    """Convert torch.Tensor in cpu to a jax array
    
    This might involves copying the data (depending if dlpack is enabled)
    """
    return torch_pytree.tree_map_only(
        jax.Array,
        lambda x: mappings.j2t(x, self.config.use_dlpack_for_data_conversion),
        args)

  def t2j_copy(self, args):
    """Convert jax array to torch.Tensor in cpu.
    
    This might involves copying the data (depending if dlpack is enabled)
    """
    return torch_pytree.tree_map_only(
        torch.Tensor,
        lambda x: mappings.t2j(x, self.config.use_dlpack_for_data_conversion),
        args)

  def override_op_definition(self, op_to_override, op_impl):
    self._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
    )
