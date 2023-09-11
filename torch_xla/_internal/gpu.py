import os
import torch_xla
import torch_xla.core.xla_env_vars as xenv

distributed_service = None


def num_local_processes() -> int:
  """Returns number of processes to create on this host.
    
  Raises:
    AssertionError: if GPU_NUM_DEVICES environment variable
                    is not configured
  """
  assert xenv.GPU_NUM_DEVICES in os.environ, \
      "Must set `GPU_NUM_DEVICES` environment variable to use the PjRt GPU client"
  return int(os.environ[xenv.GPU_NUM_DEVICES])


def initialize_distributed_runtime(world_size: int) -> None:
  """Configures GPU distributed runtime parameters.

  Must be run before using any XLA devices.

  Args:
    world_size: number of devices in the cluster.
  """
  if world_size > 1:
    # TODO(jonbolin): For multi-host, this needs to be consistent across hosts
    global distributed_service
    if distributed_service is None:
      num_nodes = world_size
      distributed_service = torch_xla._XLAC._xla_get_distributed_runtime_service(
          num_nodes)


def shutdown_distributed_runtime() -> None:
  """Destroy the distributed runtime after a distributed computation."""
  global distributed_service
  if distributed_service:
    distributed_service.shutdown()
    distributed_service = None
