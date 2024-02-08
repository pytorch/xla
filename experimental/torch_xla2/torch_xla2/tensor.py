import functools
import jax
from jax import dlpack as jaxdl
import jax.numpy as jnp
import numpy
import torch
import torch._decomp as decomp
import torch._decomp.decompositions
from torch_xla2 import ops_registry
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree
import torch.utils.dlpack as torchdl


class XLADispatchMode(torch_dispatch.TorchDispatchMode):

  def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
    if fn in constructors:
      args, kwargs = unwrap((args, kwargs))
      res = constructors[fn](*args, **kwargs)
      return wrap(res)
    return fn(*args, **kwargs)


def _aten_arange(
    start,
    end,
    *,
    dtype=None,
    layout=None,
    requires_grad=False,
    device=None,
    pin_memory=False
):
  return jnp.arange(start, end, 1)


constructors = {
    torch.ops.aten.arange.default: functools.partial(_aten_arange, 0),
    torch.ops.aten.arange.start: _aten_arange,
}


def wrap(jaxarray):
  return torch_pytree.tree_map_only(jnp.ndarray, XLATensor2, jaxarray)


def unwrap(torchtensors):
  return torch_pytree.tree_map_only(XLATensor2, lambda x: x._elem, torchtensors)


def t2j(t):
  if isinstance(t, XLATensor2):
    return t._elem
  if t.dtype == torch.bool:
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
      nparray = (
          t.detach().to(torch.float32).numpy()
      )  # numpy don't support bfloat16
    else:
      nparray = t.detach().numpy()
    res = jnp.asarray(nparray)
    if t.dtype == torch.bfloat16:
      res = res.astype(jnp.bfloat16)

  if t.dtype == torch.bool:
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


def t2j_dtype(dtype):
  return {
      torch.bfloat16: jnp.bfloat16,
      torch.double: jnp.double,
      torch.float32: jnp.float32,
      torch.half: jnp.float16,
      torch.long: jnp.int64,
      torch.int32: jnp.int32,
      torch.int16: jnp.int16,
      torch.bool: jnp.bool_,
  }.get(dtype)


def j2t_dtype(dtype):
  return {
      jnp.bfloat16: torch.bfloat16,
      jnp.double: torch.double,
      jnp.float32: torch.float32,
      jnp.float16: torch.half,
      jnp.int64: torch.long,
      jnp.int32: torch.int32,
      jnp.int16: torch.int16,
      jnp.bool_: torch.bool,
  }.get(dtype)


def move_to_device(t):
  return XLATensor2(t2j(t))


class XLATensor2(torch.Tensor):

  @staticmethod
  def __new__(cls, elem):
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

  def __init__(self, elem: jax.Array):
    super().__init__()
    self._elem = elem

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
        self._elem.shape[:start_dim] + (-1,) + self._elem.shape[end_dim:]
    )
    new_elem = jnp.reshape(self._elem, new_shape)
    return XLATensor2(new_elem)
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
    print("running...", func.name(), types)
    for a in torch_pytree.tree_flatten(args)[0]:
      if isinstance(a, XLATensor2):
        print("  ", a._elem.shape)
      else:
        print("  ", a)
    lowering = ops_registry.lowerings.lookup(func)

    if lowering is None:
        raise RuntimeError("No lowering found for", func.name())

    with XLADispatchMode():
      res = lowering(*args, **kwargs)
    print("output:")
    for a in torch_pytree.tree_flatten(res)[0]:
      if isinstance(a, XLATensor2):
        print("  ", a._elem.shape)
    debug_accuracy(func, args, kwargs, res)
    return res

  def detach(self):
    return XLATensor2(jax.lax.stop_gradient(self.jax()))

  def numpy(self) -> numpy.ndarray:
    import numpy as np

    return np.array(self._elem)

  def jax(self) -> jax.Array:
    return self._elem

  def torch(self) -> torch.Tensor:
    return j2t(self.jax())


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
      torch.Tensor, lambda x: j2t(x._elem), (args, kwargs, current_output)
  )
  expected_out = func(*args_torch, **kwargs_torch)

  flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
  flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

  for ex, real in zip(flattened_expected_out, flattened_current_out):
    if ex.dtype != real.dtype:
      ex = ex.to(real.dtype)
    try:
      if (
          _DEBUG_ACCURACY
          and isinstance(ex, torch.Tensor)
          and not torch.allclose(ex, real, atol=1e-3, equal_nan=True)
      ):
        import pdb

        pdb.set_trace()
    except:
      import pdb

      pdb.set_trace()

  return True
