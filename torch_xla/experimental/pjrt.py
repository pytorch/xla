import functools
import logging

from typing import TypeVar
from torch_xla import runtime
from torch_xla._internal import multiprocess
import torch_xla.core.xla_model as xm

FN = TypeVar('FN')


def deprecated(new: FN) -> FN:
  already_warned = [False]
  @functools.wraps(new)
  def wrapped(*args, **kwargs):
    if not already_warned[0]:
      logging.warning(f'{__name__}.{new.__name__} is deprecated. Use {new.__module__}.{new.__name__} instead.')
      already_warned[0] = True

    return new(*args, **kwargs)

  return wrapped

def register_deprecated(new: FN):
  globals()[new.__name__] = deprecated(new)


aliases = [
  runtime.addressable_device_count,
  runtime.device_attributes,
  runtime.device_type,
  runtime.global_device_count,
  runtime.global_ordinal,
  runtime.local_device_count,
  runtime.local_ordinal,
  runtime.local_process_count,
  runtime.process_count,
  runtime.process_index,
  runtime.rendezvous,
  runtime.set_device_type,
  runtime.using_pjrt,
  runtime.world_size,
  runtime.xla_device,
  multiprocess.spawn,
  xm.broadcast_master_param,
]

for a in aliases:
  register_deprecated(a)
