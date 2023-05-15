from torch_xla import runtime
from torch_xla._internal import multiprocess
from torch_xla.experimental.deprecation import register_deprecated
import torch_xla.core.xla_model as xm

from . import pjrt as this_module

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

for alias in aliases:
  register_deprecated(this_module, alias)
