import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


class AllReduce(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, reduce_type, scale, groups):
    ctx.reduce_type = reduce_type
    ctx.scale = scale
    output = xm.all_reduce(reduce_type, input, scale=scale, groups=groups)
    ctx.save_for_backward(input, output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input, output = ctx.saved_tensors
    grad = grad_output * ctx.scale if ctx.scale != 1.0 else grad_output
    if ctx.reduce_type == xm.REDUCE_SUM:
      return grad, None, None, None
    if ctx.reduce_type == xm.REDUCE_MUL:
      # MUL is not supported by TPU
      grad_scaler = torch.where(input != 0, output / input,
                                torch.zeros_like(input))
      return grad * grad_scaler, None, None, None
    if ctx.reduce_type == xm.REDUCE_MIN or ctx.reduce_type == xm.REDUCE_MAX:
      return torch.where(input == output, grad,
                         torch.zeros_like(grad)), None, None, None
    raise RuntimeError('Unsupported reduce type: {}'.format(ctx.reduce_type))


def all_reduce(reduce_type, value, scale=1.0, groups=None):
  """Performs an inplace reduce operation on the input tensor.

  This is the same as `xm.all_reduce()` but supports autograd differentiation.

  Args:
    reduce_type (string): One of ``REDUCE_SUM``, ``REDUCE_MUL``, ``REDUCE_AND``,
      ``REDUCE_OR``, ``REDUCE_MIN`` and ``REDUCE_MAX``.
    value (torch.Tensor): The to perform the all reduce op to.
    scale (float): A default scaling value to be applied after the reduce.
      Default: 1.0
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
  Returns:
    The reduced value across the selected replicas.
  """
  return AllReduce.apply(value, reduce_type, scale, groups)


class AllGather(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, dim):
    ctx.dim = dim
    ctx.ordinal = xr.global_ordinal()
    ctx.world_size = xr.world_size()
    return xm.all_gather(input, dim=dim)

  @staticmethod
  def backward(ctx, grad_output):
    slice_size = grad_output.size(ctx.dim) // ctx.world_size
    return torch.narrow(grad_output.clone(), ctx.dim, ctx.ordinal * slice_size,
                        slice_size), None


def all_gather(value, dim=0):
  """Performs an all-gather operation along a given dimension.

  This is the same as `xm.all_gather()` but supports autograd differentiation.

  Args:
    value (torch.Tensor): The input tensor.
    dim (int): The gather dimension.
      Default: 0
  Returns:
    A tensor which has, in the ``dim`` dimension, all the values from the
    participating replicas.
  """
  return AllGather.apply(value, dim)


def distributed_mm(w, x, split=1):
  """Performs a matrix multiplication with sharded weight.

  Args:
    w (torch.Tensor): The sharded weight, RHS of the matrix multiplication
      operation. The weight shape is `N x Ko` where `Ko` is the shard
      dimension size. Each ordinal will have its own copy of the weight.
    x (torch.Tensor): The input tensor, LHS of the matrix multiplication
      operation. The input shape is `WG x M` where `WG = Ko * WORLD_SIZE`.
    split (int): The number of splits for the `M` dimension of `x`. Since
      there is an `all_gather()` on such dimension, if `M` is big, a split
      might be required in order to fit device memory.
      Default: 1
  Returns:
    The result of the distributed matrix multiplication operation.
  """
  ordinal = xr.global_ordinal()
  # w = N x Ko
  # WG = Ko * WORLD_SIZE
  # x = WG x M
  assert x.size(0) // xr.world_size() == w.size(1)
  splits = []
  if split != 1:
    size = x.size(1)
    assert size % split == 0
    split_size = size // split
    splits = torch.split(x, split_size, dim=1)
  else:
    splits.append(x)
  results = []
  for xs in splits:
    # xg = WG x (M * WORLD_SIZE)
    xg = all_gather(xs, dim=1)
    # xgn = Ko x (M * WORLD_SIZE)
    xgn = torch.narrow(xg, 0, ordinal * w.size(1), w.size(1))
    # wxg = N x (M * WORLD_SIZE)
    wxg = w @ xgn
    # rwxg = N x (M * WORLD_SIZE)
    rwxg = all_reduce(xm.REDUCE_SUM, wxg)
    # wx = N x M
    wx = torch.narrow(rwxg, 1, ordinal * xs.size(1), xs.size(1))
    results.append(wx)
  return torch.cat(results, dim=1) if len(results) > 1 else results[0]


class SyncBatchNorm(torch.nn.Module):

  def __init__(
      self,
      num_features: int,
      eps: float = 1e-5,
      momentum: float = 0.1,
  ):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.weight = torch.nn.Parameter(torch.ones(num_features))
    self.bias = torch.nn.Parameter(torch.zeros(num_features))
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
    assert 2 <= batch.ndim <= 5 and batch.shape[1] == self.num_features
    reduce_dims = list(range(batch.ndim))
    reduce_dims.pop(1)  # channel dim

    if self.training:
      local_mean = torch.mean(batch, dim=reduce_dims)
      local_sqr_mean = torch.mean(batch * batch, dim=reduce_dims)

      scale = 1.0 / xr.world_size()
      mean = AllReduceSumLayer.apply(local_mean) * scale
      sqr_mean = AllReduceSumLayer.apply(local_sqr_mean) * scale
      var = sqr_mean - mean.pow(2)

      self.running_mean = (
          1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (
          1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    res = torch.empty_like(batch)
    for c in range(self.num_features):
      if batch.ndim == 2:
        res = ((batch - mean) /
               torch.sqrt(var + self.eps)) * self.weight + self.bias
      else:
        res[:, c, ...] = (
            (batch[:, c, ...] - mean[c]) /
            torch.sqrt(var[c] + self.eps)) * self.weight[c] + self.bias[c]

    return res

  def extra_repr(self) -> str:
    return f'{self.num_features}, eps={self.eps}'


class AllReduceSumLayer(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    return xm.all_reduce(xm.REDUCE_SUM, x)

  @staticmethod
  def backward(ctx, grad_output):
    return xm.all_reduce(xm.REDUCE_SUM, grad_output)
