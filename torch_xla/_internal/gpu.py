import os
import torch_xla.core.xla_env_vars as xenv


def num_local_processes() -> int:
  """Returns number of processes to create on this host.
    
  Raises:
    AssertionError: if GPU_NUM_DEVICES environment variable
                    is not configured
  """
  assert xenv.GPU_NUM_DEVICES in os.environ, \
      "Must set `GPU_NUM_DEVICES` environment variable to use the PjRt GPU client"
  return int(os.environ[xenv.GPU_NUM_DEVICES])


def initialize_env(local_world_size: int):
  os.environ[xenv.LOCAL_WORLD_SIZE] = str(local_world_size)
