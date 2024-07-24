import logging
import re
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


def parse_xla_device(device: str):
  m = re.match(r'([A-Z]+):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))


def reduce_gradients(optimizer, groups=None, pin_layout=True):
  """Reduces all the gradients handled by an optimizer.

  Args:
    optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance
      containing the gradients to be reduced.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
    pin_layout (bool, optional): whether to pin the layout when reducing gradients.
      See `xm.all_reduce` for details.
  """
  count = xm.xrt_world_size()
  if count > 1:
    gradients = _fetch_gradients(optimizer)
    bucket_cap_mb = int(os.getenv('ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB', 0))
    # Reverse the gradients list so that we start allreduce from the last layer
    # onwards. This allows allreduce to trigger as soon as the bucket fills up and
    # overlap with backward pass.
    if bucket_cap_mb > 0:
      gradients = reversed(gradients)
      xm.all_reduce_bucketized_gradients(
          gradients,
          scale=1.0 / count,
          groups=groups,
          pin_layout=pin_layout,
          bucket_cap_mb=bucket_cap_mb)
    else:
      xm.all_reduce(
          REDUCE_SUM,
          gradients,
          scale=1.0 / count,
          groups=groups,
          pin_layout=pin_layout)


class ToXlaTensorArena(object):

  def __init__(self, convert_fn, select_fn):
    self._convert_fn = convert_fn
    self._select_fn = select_fn
    self._tensors = []

  def _add(self, tensor):
    self._tensors.append(tensor)

  def _convert(self):
    self._index = 0
    if self._tensors:
      self._converted_tensors = self._convert_fn(self._tensors)
    else:
      self._converted_tensors = []

  def _get_converted_tensor(self):
    assert self._index < len(self._converted_tensors)
    new_tensor = self._converted_tensors[self._index]
    self._index += 1
    return new_tensor

  def _collect_tensors(self, inputs):

    def collect_fn(value):
      self._add(value)

    xu.for_each_instance(inputs, lambda x: self._select_fn(x), collect_fn)

  def _replace_tensors(self, inputs):

    def convert_fn(value):
      return self._get_converted_tensor()

    return xu.for_each_instance_rewrite(inputs, lambda x: self._select_fn(x),
                                        convert_fn)

  def transform(self, inputs):
    self._tensors = []
    self._collect_tensors(inputs)
    self._convert()
    return self._replace_tensors(inputs)
