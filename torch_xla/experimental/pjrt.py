from torch_xla import runtime
from torch_xla._internal import pjrt
from torch_xla.experimental.deprecation import deprecated, register_deprecated
import torch_xla.core.xla_model as xm

from . import pjrt as this_module

aliases = [
    runtime.addressable_device_count,
    runtime.device_type,
    runtime.global_device_count,
    runtime.global_ordinal,
    runtime.local_device_count,
    runtime.local_ordinal,
    runtime.local_process_count,
    runtime.process_count,
    runtime.process_index,
    runtime.set_device_type,
    runtime.using_pjrt,
    runtime.world_size,
    runtime.xla_device,
    pjrt.spawn,
    pjrt.spawn_threads,
    pjrt.run_multiprocess,
    xm.broadcast_master_param,
]

rendezvous = deprecated(this_module, xm.xla_rendezvous)
device_attributes = deprecated(this_module, runtime.runtime_device_attributes)
global_device_attributes = deprecated(this_module,
                                      runtime.global_runtime_device_attributes)

for alias in aliases:
  register_deprecated(this_module, alias)
