import os
import atexit
import torch_xla
import torch_xla.core.xla_env_vars as xenv

distributed_service = None


def num_local_processes() -> int:
  """Returns number of processes to create on this host.
    
  Raises:
    AssertionError: if GPU_NUM_DEVICES environment variable
                    is not configured
  """
  assert xenv.WORLD_SIZE in os.environ, \
      "Must set `WORLD_SIZE` environment variable to use the PjRt GPU client"
  return int(os.environ[xenv.WORLD_SIZE])


def initialize_distributed_runtime(global_world_size: int) -> None:
  """Configures GPU distributed runtime parameters.

  Must be run before using any XLA devices.

  Args:
    global_world_size: number of devices in the cluster.
  """
  if global_world_size > 1:
    global distributed_service
    if distributed_service is None:
      num_nodes = global_world_size
      distributed_service = torch_xla._XLAC._xla_get_distributed_runtime_service(
          num_nodes)
      atexit.register(shutdown_distributed_runtime)


def shutdown_distributed_runtime() -> None:
  """Destroy the distributed runtime after a distributed computation."""
  global distributed_service
  if distributed_service:
    print('xw32 shutdown_distributed_runtime: shutdown begins')
    distributed_service.shutdown()
    distributed_service = None
    print('xw32 shutdown_distributed_runtime: shutdown finishes')
