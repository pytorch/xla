import torch
import torch_xla
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
      ``REDUCE_OR``, ``REDUCE_MIN`` and ``REDUCE_MIN``.
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
    ctx.ordinal = xm.get_ordinal()
    ctx.world_size = xm.xrt_world_size()
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


def nms(boxes, scores, score_threshold, iou_threshold, output_size):
  """Performs a Non Maximal Suppression operation.

  Args:
    boxes (torch.Tensor): A `torch.Tensor` of shape `[N, 4]` listing the boxes
      coordinates in `(y0, x0, y1, x1)` form.
    scores (torch.Tensor): A `torch.Tensor` of shape `[N]` listing the scores
      of each box.
    score_threshold (torch.Tensor): The minimum score for a box to qualify as
      valid.
    iou_threshold (torch.Tensor): The minimum IOU (Intersection Over Union)
      score to trigger overlap logic.
    output_size (int): The maximum number of returned indices (must be lower or
      equal to N).

  Returns:
    A tuple of `torch.Tensor` with the first element being the selected box
    indices, and the second element being the number of valid boxes.
  """
  return torch_xla._XLAC._xla_nms(boxes, scores, score_threshold, iou_threshold,
                                  output_size)
