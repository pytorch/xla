import os
from torch_xla.experimental import plugins
import torch_xla.utils.utils as xu


class RocmPlugin(plugins.DevicePlugin):
  def _get_process_rank(self) -> int:
    local_process_rank = xu.getenv_as("PJRT_LOCAL_PROCESS_RANK", int,
                                      xu.getenv_as("LOCAL_RANK", int, 0))
    global_process_rank = xu.getenv_as("RANK", int, local_process_rank)

    return local_process_rank, global_process_rank

  def _get_world_size(self) -> int:
    local_world_size = xu.getenv_as("PJRT_LOCAL_PROCESS_COUNT", int,
                                    xu.getenv_as("LOCAL_WORLD_SIZE", int, 1))
    global_world_size = xu.getenv_as("WORLD_SIZE", int, local_world_size)

    return local_world_size, global_world_size
  
  def library_path(self) -> str:
    return os.path.join(os.path.dirname(__file__), 'lib', 'pjrt_c_api_gpu_plugin.so')

  def physical_chip_count(self) -> int:
    # TODO: default to actual device count
    return xu.getenv_as('GPU_NUM_DEVICES', int, 1)
  
  def client_create_options(self) -> dict:
    local_process_rank, global_process_rank = self._get_process_rank()
    local_world_size, global_world_size = self._get_world_size()

    # The available options are defined in OpenXLA: https://github.com/openxla/xla/blob/1bb2a74be91fabf5f9aa2702b2592b5b022c9052/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L58-L67
    options = {
        "platform_name":
            "gpu",
        # TODO(wcromar): make this configurable
        "allocator":
            "default",
        "memory_fraction":
            xu.getenv_as("PJRT_ALLOCATOR_FRACTION", float, None),
        "preallocate":
            xu.getenv_as("PJRT_ALLOCATOR_PREALLOCATE", bool, None),
        # Use all devices by default and when using SPMD
        "visible_devices": [local_process_rank]
                           if local_world_size > 1 else None,
        "node_id":
            global_process_rank,
        "num_nodes":
            global_world_size,
    }

    return {k: v for k, v in options.items() if v is not None}

  def requires_xla_coordinator(self) -> bool:
    _, global_world_size = self._get_world_size()
    return global_world_size > 1