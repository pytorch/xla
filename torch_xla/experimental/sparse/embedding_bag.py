from enum import IntEnum
from typing import Optional, Tuple
import warnings
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import sparse
from torch.autograd import Function
from .coo import SparseCOOTensor

_NativeEmbeddingModule = sparse.Embedding
_NativeEmbeddingBagModule = sparse.EmbeddingBag


class _EmbeddingBagMode(IntEnum):
  SUM = 0
  MEAN = 1
  MAX = 2


def _tensor_factory_kwargs(t):
  return {
      k: getattr(t, k)
      for k in {'dtype', 'device', 'layout', 'requires_grad', 'pin_memory'}
  }


def _apply_bag_size_backward(mode: _EmbeddingBagMode, index_grad: Tensor,
                             offset2bag: Tensor, bag_size: Tensor) -> Tensor:
  if mode == _EmbeddingBagMode.MEAN:
    index_grad *= (1 / bag_size.to(
        **_tensor_factory_kwargs(index_grad)).unsqueeze(1).index_select(
            0, offset2bag))
  return index_grad


def _sparse_embedding_backward(grad: Tensor, indices: Tensor, num_weights: int,
                               padding_idx: int):
  if padding_idx != -1:
    c = indices != padding_idx
    indices = indices.index(c)
    grad = grad.index(c)

  num_features = grad.size(-1)
  weight_size = (num_weights, num_features)
  dense_options = _tensor_factory_kwargs(grad)

  if grad.numel() == 0:
    sparse_index = torch.empty((1, 0),
                               **_tensor_factory_kwargs(indices),
                               dtype=torch.int64)
    sparse_values = torch.empty((0, num_features), **dense_options),
  else:
    sparse_index = indices.reshape(1, -1)
    sparse_values = grad.reshape((-1, num_features))

  grad_weight = SparseCOOTensor.create(sparse_index, sparse_values, weight_size)

  return grad_weight


class SparseEmbeddingBag(Function):
  """Alternate embeding bag implementation. Produces sparse gradients using an XLA compatible tensor subclass
  """

  @staticmethod
  def forward(
      weight: Tensor,
      indices: Tensor,
      offsets: Tensor,
      max_norm: Optional[float] = None,
      scale_grad_by_freq: bool = False,
      mode: _EmbeddingBagMode = _EmbeddingBagMode(0),
      sparse: bool = False,
      per_sample_weights: Optional[Tensor] = None,
      include_last_offset: bool = False,
      padding_idx: Optional[int] = None,
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # otherwise we shouldn not be here
    assert sparse
    assert weight.requires_grad
    torch._check(
        not scale_grad_by_freq,
        "embedding_bag: scale_grad_by_freq is not supported with sparse gradients"
    )
    torch._check(
        max_norm is None,
        "Itermediate renormalization is not supported with sparse=True")
    # output, offset2bag, bag_size, max_indices
    # note: autograd is disabled while we are inside here, so we won't triger
    # any double-backward behavior. We are not modifying the forward in any way
    return torch.embedding_bag(weight, indices, offsets, scale_grad_by_freq,
                               mode.value, sparse, per_sample_weights,
                               include_last_offset, padding_idx)

  @staticmethod
  def setup_context(ctx,
                    inputs: Tuple[Tensor, Tensor, Optional[Tensor],
                                  Optional[Tensor], bool, _EmbeddingBagMode,
                                  bool, Optional[Tensor], bool, Optional[int]],
                    outputs: Tuple[Tensor, Tensor, Tensor, Tensor]):
    weight, indices, offsets, _, _, mode, _, per_sample_weights, _, padding_idx = inputs
    _, offset2bag, bag_size, max_indices = outputs
    # for completness, not technically required as integer dtype will make this automatic
    ctx.mark_non_differentiable(offset2bag, bag_size, max_indices)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(indices, offsets, offset2bag, bag_size,
                          per_sample_weights)
    ctx.num_weights = weight.size(0)
    ctx.padding_idx = padding_idx
    ctx.mode = mode
    # Todo: remove after validating assumptions
    ctx.n_out = len(outputs)

  @staticmethod
  def backward(ctx, grad_: Tensor, *args: Tuple[Optional[Tensor], ...]):
    # All *args will be none, autograd will need those slots but never materialized grads for non-differentable types
    indices, offset2bag, bag_size, per_sample_weights = ctx.saved_tensors
    # we need a grad slot in the return for every tensor input to forward
    grad_weight = grad_indices = grad_offsets = grad_per_sample_weights = None
    if grad_ is not None:
      # compute grad_weight
      index_grad = grad_.index_select(0, offset2bag)
      index_grad = _apply_bag_size_backward(ctx.mode, index_grad, offset2bag,
                                            bag_size)
      if per_sample_weights is not None:
        torch._check(ctx.mode == _EmbeddingBagMode.SUM)
        index_grad.mul_(per_sample_weights.unsqueeze(1))

      grad_weight = _sparse_embedding_backward(index_grad, indices,
                                               ctx.num_weights, ctx.padding_idx)
      return grad_weight, grad_indices, grad_offsets, grad_per_sample_weights


class SparseEmbedding(Function):

  @staticmethod
  def forward(input: Tensor,
              weight: Tensor,
              padding_idx: Optional[int],
              scale_grad_by_freq: bool = False,
              sparse: bool = False) -> Tensor:
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq,
                           sparse)

  @staticmethod
  def setup_context(ctx, inputs, output):
    indices, weight, padding_idx, scale_grad_by_freq, sparse = inputs
    ctx.mark_non_differentiable(indices)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(indices)
    ctx.num_weights = weight.size(0)
    ctx.padding_idx = padding_idx

  @staticmethod
  def backward(ctx, grad_: Tensor):
    grad_input = grad_weight = None
    indices = ctx.saved_tensors[0]
    if grad_ is not None:
      grad_weight = _sparse_embedding_backward(grad_, indices, ctx.num_weights,
                                               ctx.padding_idx)

    return grad_input, grad_weight, None, None, None


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = 'mean',
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
  f"""This is the equivalant API for F.embedding_bag, but where sparse gradients for :attr:`weight` would be produced a custom path is taken.
  {F.embedding_bag.__doc__}
  """
  if weight.dtype == torch.long and input.is_floating_point():
    warnings.warn("Argument order of nn.functional.embedding_bag was changed. "
                  "Usage `embedding_bag(weight, input, ...)` is deprecated, "
                  "and should now be `embedding_bag(input, weight, ...)`.")
    weight, input = input, weight

  if per_sample_weights is not None and input.size() != per_sample_weights.size(
  ):
    raise ValueError(
        f"embedding_bag: If per_sample_weights ({per_sample_weights.shape}) is not None, "
        f"then it must have the same shape as the input ({input.shape})")

  if not weight.dim() == 2:
    raise ValueError(
        f"weight has to be a 2D Tensor, but got Tensor of dimension {weight.dim()}"
    )

  if not torch.jit.is_scripting() and input.dim() == 2 and input.is_nested:
    include_last_offset = True
    offsets = input.offsets()
    input = input.values().reshape(-1)
    if per_sample_weights is not None:
      if not per_sample_weights.is_nested:
        raise ValueError(
            "If input is nested, then per_sample_weights must be nested if specified"
        )
      per_sample_weights = per_sample_weights.values().reshape(-1)
  elif input.dim() == 2:
    if offsets is not None:
      type_str = "<unknown>"
      # TODO: Remove this once script supports type() calls
      if not torch.jit.is_scripting():
        type_str = str(type(offsets))
      raise ValueError("if input is 2D, then offsets has to be None"
                       ", as input is treated is a mini-batch of"
                       " fixed length sequences. However, found "
                       f"offsets of type {type_str}")
    offsets = torch.arange(
        0, input.numel(), input.size(1), dtype=input.dtype, device=input.device)

    input = input.reshape(-1)
    if per_sample_weights is not None:
      per_sample_weights = per_sample_weights.reshape(-1)
  elif input.dim() == 1:
    if offsets is None:
      raise ValueError("offsets has to be a 1D Tensor but got None")
    if offsets.dim() != 1:
      raise ValueError("offsets has to be a 1D Tensor")
  else:
    raise ValueError(
        f"input has to be 1D or 2D Tensor, but got Tensor of dimension {input.dim()}"
    )
  if mode == "sum":
    mode_enum = _EmbeddingBagMode.SU
  elif mode == "mean":
    mode_enum = _EmbeddingBagMode.MEAN
  elif mode == "max":
    mode_enum = _EmbeddingBagMode.MAX

    if scale_grad_by_freq:
      raise ValueError(
          "max mode does not support scaling the gradient by the frequency")

    if sparse:
      raise ValueError("max mode does not support sparse weights")

  else:
    raise ValueError("mode has to be one of sum, mean or max")

  if max_norm is not None:
    with torch.no_grad():
      # modified weight in place!
      torch.embedding_renorm_(weight, input, max_norm, norm_type)

  if per_sample_weights is not None and mode != "sum":
    raise NotImplementedError(
        "embedding_bag: per_sample_weights was not None. "
        "per_sample_weights is only supported for mode='sum' "
        f"(got mode='{mode}'). Please open a feature request on GitHub.")

  impl = SparseEmbeddingBag.apply if sparse and weight.requires_grad(
  ) else F.embedding_bag

  ret, _, _, _ = impl(
      weight,
      input,
      offsets,
      scale_grad_by_freq,
      mode_enum,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx,
  )
  return ret


def embedding(input: Tensor,
              weight: Tensor,
              padding_idx: Optional[int] = None,
              max_norm: Optional[float] = None,
              norm_type: float = 2.0,
              scale_grad_by_freq: bool = False,
              sparse: bool = False) -> Tensor:
  f"""This is the equivalant API for F.embedding_bag, but where sparse gradients for :attr:`weight` would be produced a custom path is taken.
  {F.embedding.__doc__}
  """
  if padding_idx is not None:
    if padding_idx > 0:
      assert padding_idx < weight.size(
          0), "Padding_idx must be within num_embeddings"
    elif padding_idx < 0:
      assert padding_idx >= -weight.size(
          0), "Padding_idx must be within num_embeddings"
      padding_idx = weight.size(0) + padding_idx
  else:
    padding_idx = -1
  if max_norm is not None:
    # Note [embedding_renorm contiguous]
    # `embedding_renorm_` will call .contiguous() on input anyways, so we
    # call it here and take advantage of the improved locality in the
    # `embedding` call below too.
    input = input.contiguous()
    # Note [embedding_renorm set_grad_enabled]
    # Note [Modified from F.embedding]
    # We do't need to support torch-script so we can just do what they say this is equivalant to, original below
    #_no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    with torch.no_grad():
      torch.embedding_renorm_(weight, input, max_norm, norm_type)

  impl = SparseEmbedding.apply if sparse and weight.requires_grad else torch.embedding
  return impl(input, weight, padding_idx, scale_grad_by_freq, sparse)


class EmbeddingBag(_NativeEmbeddingBagModule):
  f"""Torch-XLA backend compatible Embedding Bag
  
  This implementation modifies the forward function when :attr:`sparse` is ``True``.
  The alternate implementation has a backward which will produce sparse
  gradients w.r.t :attr:`weight` as a tensor subclass implemention of the
  ``torch.sparse_coo`` layout. If :attr:`sparse` is ``False`` this is identical
  to :class:`~torch.nn.EmbedingBag`.

  {_NativeEmbeddingBagModule.__doc__}
  """

  def forward(self,
              input: Tensor,
              offsets: Optional[Tensor] = None,
              per_sample_weights: Optional[Tensor] = None) -> Tensor:
    # setting sparse without requiring weight grads has no effect
    if self.weight.requires_grad() and self.sparse:
      return embedding_bag(input, self.weight, offsets, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.mode,
                           self.sparse, per_sample_weights,
                           self.include_last_offset, self.padding_idx)
    return super().forward(input, offsets, per_sample_weights)


class Embedding(_NativeEmbeddingModule):
  f"""Torch-XLA backend compatible Embedding 
  
  This implementation modifies the forward function when :attr:`sparse` is ``True``.
  The alternate implementation has a backward which will produce sparse
  gradients w.r.t :attr:`weight` as a tensor subclass implemention of the
  ``torch.sparse_coo`` layout. If :attr:`sparse` is ``False`` this is identical
  to :class:`~torch.nn.Embeding`.

  {_NativeEmbeddingBagModule.__doc__}
  """

  def forward(self, input: Tensor) -> Tensor:
    if self.weight.requires_grad and self.sparse:
      return embedding(input, self.weight, self.padding_idx, self.max_norm,
                       self.norm_type, self.scale_grad_by_freq, self.sparse)
    return super().forward(input)
