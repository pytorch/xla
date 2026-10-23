import torch
from torch.autograd import Function

aten = torch.ops.aten

_COO_DISPATCH_TABLE = {}

from .coo import SparseCOOTensor


def _register_dispatch_func(op):

  def wrapper(f):
    _COO_DISPATCH_TABLE[op] = f
    return f

  return wrapper


def _flatten_indices(inds, shape):
  # Flatted N-D indices to 1-D indices
  flat_indices = inds.new_zeros(inds.size(1))
  for d, sz in enumerate(shape):
    flat_indices = (flat_indices * sz) + torch.select_copy(inds, 0, d)
  return flat_indices


def _check_no_kwargs(kwargs, name):
  torch._check(
      kwargs is None or len(kwargs) == 0,
      lambda: f"{name}: expected no kwargs, got {kwargs}")


def _check_strided(arg, f_name, arg_name):
  torch._check(
      arg.layout == torch.strided,
      lambda: f"{f_name}: Expected strided {arg_name}, not {arg.layout}")


def _check_sparse(arg, f_name, arg_name):
  torch._check(
      type(arg) is SparseCOOTensor,
      lambda: f"{f_name}: Expected sparse {arg_name}, not {arg.layout}")


class _SparseMaker(Function):

  @staticmethod
  def forward(ctx, indices, values, size, dtype=None, device=None):
    return SparseCOOTensor(indices, values, size, dtype=dtype, device=device)

  @staticmethod
  def backward(ctx, *args):
    # args will be none except values which we want to send through
    return args


def make_sparse(indices, values, size, dtype=None, device=None):
  return _SparseMaker.apply(indices, values, size, dtype, device)


@_register_dispatch_func(aten.sparse_mask)
def coo_sparse_mask(args=(), kwargs=None):
  """
    x.sparse_mask(y)
    create a new sparse tensor from strided x and sparse y.
    Result holds values from x with sparsity pattern of Y
    """
  torch._check(
      len(args) == 2,
      lambda: f"sparse_mask: Expected two arguments, got {len(args)}")
  self, mask = args
  _check_strided(self, "sparse_mask", "self")
  _check_sparse(mask, "sparse_mask", "mask")
  torch._check(
      mask.size() == self.size(), lambda:
      f"sparse_mask: expected mask and self to have the same shape (self: {self.size()}, mask: {mask.size()})"
  )
  mask = mask.coalesce()
  assert mask._nnz() == mask._v.shape[0]
  assert mask._nnz() == mask._i.shape[-1]
  _check_no_kwargs(kwargs, "sparse_mask")
  mask_indices = mask._indices()
  sparse_size = mask.shape[:mask.sparse_dim()]
  dense_size = mask.shape[mask.sparse_dim():]
  flat_indices = _flatten_indices(mask_indices, sparse_size)
  # flatten out the sparse dims, which we use the flattened indices to select along that dimension
  # leave the dense dims intact so that values has the correct shape
  values = self.view((-1,) + dense_size)[flat_indices]

  return make_sparse(
      mask_indices,
      values,
      size=self.size(),
      dtype=values.dtype,
      device=values.device)


@_register_dispatch_func(aten.sparse_dim)
def coo_sparse_dim(args=(), kwargs=None):
  self = args[0]
  _check_sparse(self, "sparse_dim", "self")
  return self._i.shape[0]


@_register_dispatch_func(aten.resize_as_)
def coo_resize_as_(args, kwargs=None):
  # The aten impl supports more cases, but the only one we need is the 0-nnz case where we can freely modify the sparse/dense dims at will.
  # We have also not implemented dense dim consideration anywhere as they are not needed.

  torch._check(len(args) == 2, f"resize_as_: expected two args not {len(args)}")
  self, other = args
  _check_sparse(self, "resize_as_", "self")
  _check_sparse(other, "resize_as_", "other")
  torch._check(
      self._nnz() == 0, lambda:
      "resize_as_: resizing a sparse tensor with nnz != 0 is not supported")
  _check_no_kwargs(kwargs, "resize_as_")

  new_nnz = other._nnz()
  values_shape = (new_nnz,)
  index_shape = (len(other.shape), new_nnz)

  # attribute access to modify the tensor in-place, accessors return a clone
  self._v.resize_(values_shape)
  self._i.resize_(index_shape)
  return self


@_register_dispatch_func(aten.is_coalesced)
def coo_is_coalesced(args, kwargs=None):
  _check_no_kwargs(kwargs, "is_coalesced")
  self = args[0]
  _check_sparse(self, "is_coalesced", "self")
  return self._is_coalesced or self._nnz() < 2


@_register_dispatch_func(aten.coalesce)
def coo_coalesce(args, kwargs=None):
  _check_no_kwargs(kwargs, "coalesce")
  self = args[0]
  _check_sparse(self, "coalesce", "self")
  if self.is_coalesced:
    return self
  return torch._coalesce(self)


@_register_dispatch_func(aten._coalesce)
def coo__coalesce(args, kwargs=None):
  self = args[0]

  indices = self._indices()
  sparse_dim = self.sparse_dim()
  torch._check(sparse_dim == 1,
               lambda: "coalesce: implementation supports sparse_dim=1 only.")
  sparse_shape = self.shape[:sparse_dim]
  dense_shape = self.shape[sparse_dim:]
  indices_scalar = _flatten_indices(indices, sparse_shape)
  new_indices = indices_scalar.unique()
  old_values = self._values()
  new_nnz = new_indices.shape[0]
  new_val_elements = tuple(old_values[new_indices[i] == indices_scalar].sum(0)
                           for i in range(new_nnz))
  r = make_sparse(
      new_indices.unsqueeze(0), torch.stack(new_val_elements, dim=0),
      self.shape)
  r.is_coalesced = True
  return r


@_register_dispatch_func(aten.add_)
def coo_add_(args=(), kwargs=None):
  alpha = kwargs.pop('alpha', 1)
  torch._check(
      len(args) == 2, lambda: f"add_: expected two operands, got {len(args)}")
  self, other = args
  _check_sparse(self, "self", "add_")
  _check_sparse(other, "other", "add_")

  # todo: I think this is going to satisfy the needs for sparseADAM opt
  torch._check(
      self._i.equal(other._i).all(),
      "add_: Operation is supported when operands have the same sparsity pattern."
  )
  self._v += alpha * other._v
  return self


@_register_dispatch_func(aten.add)
def coo_add(args=(), kwargs=None):
  alpha = kwargs.pop('alpha', 1)
  torch._check(
      len(args) == 2, lambda: f"add: expected two operands, got {len(args)}")
  self, other = args
  _check_sparse(other, "other", "add")

  out = self.clone()
  out[other._indices().squeeze(0)] += (alpha * other._values())
  return out


@_register_dispatch_func(aten.indices)
def coo_indices(args=(), kwargs=None):
  torch._check(len(args) == 1, "indices: expected one argument")
  self = args[0]
  _check_sparse(self, "indices", "self")
  torch._check(self.is_coalesced, lambda: "indices: input must be coalesced")
  return self._i.clone()


@_register_dispatch_func(aten._indices)
def coo_indices_unsafe(args=(), kwargs=None):
  self = args[0]
  return self._i


@_register_dispatch_func(aten._values)
def coo_values_unsafe(args=(), kwargs=None):
  self = args[0]
  return self._v


@_register_dispatch_func(aten.values)
def coo_values(args=(), kwargs=None):
  torch._check(len(args) == 1, "values: expected one argument")
  self = args[0]
  _check_sparse(self, "values", "self")
  torch._check(self._is_coalesced, "values: input must be coalesced")
  return self._v.clone()


@_register_dispatch_func(aten._nnz)
def coo_nnz(args=(), kwargs=None):
  torch._check(len(args) == 1, "values: expected one argument")
  self = args[0]
  _check_sparse(self, "nnz", "self")
  return self._v.shape[0]


@_register_dispatch_func(aten.clone)
def coo_clone(args=(), kwargs=None):
  torch._check(len(args) == 1, "clone: expected one argument")
  self = args[0]
  return make_sparse(self._i.clone(), self._v.clone(), size=self.shape)


@_register_dispatch_func(aten.detach)
def coo_detach(args=(), kwargs=None):
  torch._check(len(args) == 1, lambda: "detach: expected 1 argument")
  _check_no_kwargs(kwargs, "detach")
  self = args[0]
  return make_sparse(self._i, aten.detach(self._v), size=self.shape)


@_register_dispatch_func(aten._to_dense)
def coo_to_dense(args=(), kwargs=None):
  masked_grad = kwargs.pop('masked_grad', False)
  torch._check(
      not masked_grad, lambda:
      "to_dense: Masked gradients are not supported for to_dense with xla sparse tensor"
  )
  dtype = kwargs.pop('dtype', None)
  torch._check(not dtype,
               lambda: "to_dense: to_dense does not support the dtype kwarg.")
  device = kwargs.get('device', self.device)
  torch._check(len(args) == 1, lambda: "to_dense: expected 1 argument")
  self = args[0]
  out = torch.zeros(self.size, dtype=self.dtype, device=device)
  inds = self.indices()
  vals = self.values()
  out[inds.t()] = vals
  return out


@_register_dispatch_func(aten._to_copy)
def coo_to_copy(args=(), kwargs=None):
  self = args[0]
  _check_sparse(self, "_to_copy", "self")
  v_dtype = kwargs.get('dtype', self._v.dtype)
  device = kwargs.get('device', self._v.device)
  new_i = torch.empty(self._i.shape, dtype=self._i.dtype, device=device)
  new_v = torch.empty(self._v.shape, dtype=v_dtype, device=device)
  new_i.copy_(self._i)
  new_v.copy_(self._v)
  return make_sparse(new_i, new_v, self.shape)


@_register_dispatch_func(aten.to)
def coo_to(args=(), kwargs=None):
  self = args[0]
  _check_sparse(self, "to", "self")
  dtype = kwargs.get('dtype', self.dtype)
  device = kwargs.get('device', self.device)
  layout = kwargs.get('layout', self.layout)
  if layout != self.layout:
    torch._check(layout == torch.strided,
                 lambda: f"to: only layout conversion to strided is supported")
    # .to(dtype) is no op if dtype is not changed
    ret = torch.to_dense(self)
  else:
    need_copy = dtype != self.dtype or device != self.device or layout != self.layout
    copy = kwargs.get('copy', need_copy)
    non_blocking = kwargs.get('non_blocking', False)
    if copy:
      # otherwise ignore non_blocking
      torch._check(not non_blocking,
                   lambda: f"to: sparse_coo does not support async copy")
      ret = torch._to_copy(
          args=(self,), kwargs={
              'dtype': dtype,
              'device': device
          })
    else:
      # if no conversion is needed to returns an alias to this
      ret = self
    return ret
