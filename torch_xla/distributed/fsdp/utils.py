import torch
import torch_xla.core.xla_model as xm


def dummy_all_gather(value, dim=0):
  repeat_num = [1] * value.dim()
  repeat_num[dim] = xm.xrt_world_size()
  return value.repeat(tuple(repeat_num))


def dummy_reduce_scatter(reduce_type, input, scale, scatter_dim, shard_count):
  full_size = input.size(scatter_dim)
  shard_size = full_size // xm.xrt_world_size()
  begin = shard_size * xm.get_ordinal()
  end = begin + shard_size
  slices = [None] * input.dim()
  slices[scatter_dim] = slice(begin, end)
  return input[tuple(slices)]
