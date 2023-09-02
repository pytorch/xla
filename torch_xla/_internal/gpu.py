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


def num_global_processes() -> int:
  """Returns number of processes to create.
    
  Raises:
    AssertionError: if GPU_NUM_DEVICES environment variable
                    is not configured
  """
  assert xenv.GPU_NUM_DEVICES_GLOBAL in os.environ, \
      "Must set `GPU_NUM_DEVICES_GLOBAL` environment variable to use the PjRt GPU client"
  return int(os.environ[xenv.GPU_NUM_DEVICES_GLOBAL])


def initialize_distributed_runtime(local_world_size: int, local_rank: int) -> None:
  """Configures GPU distributed runtime parameters.

  Must be run before using any XLA devices.

  Args:
    local_world_size: number of devices in the cluster.
  """
  print('xw32 initialize_distributed_runtime begins. local_world_size=', local_world_size, ', local_rank=', local_rank)
  # TODO(xiowei): for single host, users don't need to set MASTER_ADDR and the localhost should be used.
  assert 'GPU_NUM_DEVICES_GLOBAL' in os.environ, \
      "Must set `GPU_NUM_DEVICES_GLOBAL` environment variable to use the PjRt GPU client" 
  if 'GPU_NUM_DEVICES_GLOBAL' in os.environ:
    multi_host = True
    global_world_size = int(os.environ['GPU_NUM_DEVICES_GLOBAL'])
    assert 'HOST_RANK' in os.environ, \
        "Must set `HOST_RANK` environment variable to use the PjRt GPU client" 
    host_rank = int(os.environ['HOST_RANK'])
    assert xenv.PJRT_DIST_SERVICE_ADDR in os.environ, \
      "Must set `PJRT_DIST_SERVICE_ADDR` environment variable to use the PjRt GPU client" 
  else:
    multi_host = False
    os.environ.setdefault(xenv.PJRT_DIST_SERVICE_ADDR, '127.0.0.1:8547')

  if (not multi_host and local_world_size > 1 and local_rank == 0) or (multi_host and global_world_size > 1 and host_rank == 0 and local_rank == 0):
    # TODO(jonbolin): For multi-host, this needs to be consistent across hosts
    # TODO(xiowei): remove the below temperary check later.
    # os.environ.setdefault(xenv.PJRT_DIST_SERVICE_ADDR, '127.0.0.1:8547')
    print('xw32 gpu.py initialize_distributed_runtime: os.environ[xenv.PJRT_DIST_SERVICE_ADDR]=', os.environ[xenv.PJRT_DIST_SERVICE_ADDR])
    global distributed_service
    if distributed_service is None:
      num_nodes = global_world_size
      distributed_service = torch_xla._XLAC._xla_get_distributed_runtime_service(
          num_nodes)


def shutdown_distributed_runtime() -> None:
  """Destroy the distributed runtime after a distributed computation."""
  global distributed_service
  if distributed_service:
    distributed_service.shutdown()
    distributed_service = None
