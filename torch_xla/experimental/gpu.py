from torch_xla.experimental.deprecation import register_deprecated
from torch_xla._internal import gpu

from . import gpu as this_module

aliases = [
    gpu.initialize_distributed_runtime,
    gpu.num_local_processes,
    gpu.shutdown_distributed_runtime,
]

for alias in aliases:
  register_deprecated(this_module, alias)
