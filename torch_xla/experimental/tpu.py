from torch_xla.experimental.deprecation import register_deprecated
from torch_xla._internal import tpu

from . import tpu as this_module

aliases = [
    tpu.build_tpu_env_from_vars,
    tpu.configure_one_chip_topology,
    tpu.configure_topology,
    tpu.discover_master_worker_ip,
    tpu.get_tpu_env,
    tpu.get_worker_ips,
    tpu.num_available_chips,
    tpu.num_available_devices,
    tpu.num_local_processes,
    tpu.num_logical_cores_per_chip,
    tpu.process_bounds_size,
    tpu.task_id,
    tpu.version,
]

for alias in aliases:
  register_deprecated(this_module, alias)
