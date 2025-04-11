from typing import Tuple
import torch


def _flatten_indices(inds, shape):
  # Flatted N-D indices to 1-D indices
  flat_indices = inds.new_zeros(inds.size(1))
  for d, sz in enumerate(shape):
    flat_indices.mul_(sz)
    flat_indices.add_(inds[d])
  return flat_indices


def _check_no_kwargs(kwargs, name):
  torch._check(
      kwargs is None or len(kwargs) == 0,
      f"{name}: expected no kwargs, got {kwargs}")


def _check_strided(arg, f_name, arg_name):
  torch._check(arg.layout == torch.strided,
               f"{f_name}: Expected strided {arg_name}, not {arg.layout}")


def _check_sparse(arg, f_name, arg_name):
  torch._check(arg.layout == torch.sparse_coo,
               f"{f_name}: Expected sparse {arg_name}, not {arg.layout}")


def coo_sparse_mask(args=(), kwargs=None):
  """
    x.sparse_mask(y)
    create a new sparse tensor from strided x and sparse y.
    Result holds values from x with sparsity pattern of Y
    """
  torch._check(
      len(args) == 2, f"sparse_mask: Expected two arguments, got {len(args)}")
  self, mask = args
  _check_strided(self, "sparse_mask", "self")
  _check_sparse(mask, "sparse_mask", "mask")
  torch._check(
      mask.size() == self.size(),
      f"sparse_mask: expected mask and self to have the same shape (self: {self.size()}, mask: {mask.size()})"
  )
  _check_no_kwargs(kwargs, "sparse_mask")
  mask_indices = mask.indices()
  flat_indices = _flatten_indices(mask_indices, mask.size())
  values = self.view(-1)[flat_indices]
  # acess the subclass ctor without a circular import!
  return mask.__class__(
      mask_indices,
      values,
      size=self.size(),
      dtype=values.dtype(),
      device=values.device(),
      requires_grad=values.requires_grad(),
  )


def coo_resize_as_(args, kwargs=None):
  # The aten impl supports more cases, but the only one we need is the 0-nnz case where we can freely modify the sparse/dense dims at will.
  # We have also not implemented dense dim consideration anywhere as they are not needed.

  torch._check(len(args) == 2, f"resize_as_: expected two args not {len(args)}")
  self, other = args
  _check_sparse(self, "resize_as_", "self")
  _check_sparse(other, "resize_as_", "other")
  torch._check(
      self.nnz() == 0,
      "resize_as_: resizing a sparse tensor with nnz != 0 is not supported")
  _check_no_kwargs(kwargs, "resize_as_")

  new_nnz = other.nnz()
  values_shape = (new_nnz,)
  index_shape = (len(other.shape), new_nnz)

  # attribute access to modify the tensor in-place, accessors return a clone
  self._v.resize_(values_shape)
  self._i.resize_(index_shape)
  return self


def coo_coalesce(args, kwargs=None):
  _check_no_kwargs(kwargs, "coalesce")
  self = args[0]
  _check_sparse(self, "coalesce", "self")
  if self._is_coalesced:
    return self
  if self.nnz() < 2:
    self._is_coalesced = True
    return self

  indices = self._i
  values = self._v
  sparse_dim = len(self.shape)
  nnz = self.nnz()
  indices_scalar = _flatten_indices(indices, self.shape)

  new_indices = torch.empty_like(indices)
  new_values = torch.empty_like(values)
  indices_buffer, indices_perm = indices_scalar.sort(0)
  i = 0
  for j in range(nnz):
    pos = indices_perm[j]
    curr = indices_buffer[j]
    if pos != curr:
      i += 1
      for d in range(sparse_dim):
        new_indices[d][i] = indices[d][pos]
    if values.numel() > 0:
      new_values[i] += values[pos]

  return self.__class__(
      new_indices,
      new_values,
      self.shape,
      dtype=self.dtype,
      device=self.device,
      requires_grad=self.requires_grad())


def coo_add_(args=(), kwargs=None):
  alpha = kwargs.pop('alpha', 1)
  torch._check(len(args) == 2, f"add_: expected two operands, got {len(args)}")
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


def coo_indices(args=(), kwargs=None):
  torch._check(len(args) == 1, "indices: expected one argument")
  self = args[0]
  _check_sparse(self, "indices", "self")
  torch._check(self._coalesced, "indices: input must be coalesced")
  return self._i.clone()


def coo_values(args=(), kwargs=None):
  torch._check(len(args) == 1, "values: expected one argument")
  self = args[0]
  _check_sparse(self, "values", "self")
  torch._check(self._coalesced, "values: input must be coalesced")
  return self._v.clone()


def coo_nnz(args=(), kwargs=None):
  torch._check(len(args) == 1, "values: expected one argument")
  self = args[0]
  _check_sparse(self, "nnz", "self")
  return self._v.numel()
