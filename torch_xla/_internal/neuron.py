import os
import logging

from torch_xla.experimental import plugins

import sys
import torch.distributed as dist

from .neuron_utils import get_visible_cores_list, remap_visible_cores

logging.basicConfig()
logger = logging.getLogger(__name__)


# Set root communication address/port
def set_rt_root_comm_id():
  if os.environ.get('NEURON_RT_ROOT_COMM_ID', None) is None:
    if 'MASTER_ADDR' not in os.environ:
      logging.warning(
          "MASTER_ADDR environment variable is not set, defaulting to localhost"
      )
    root_port = 62182
    root_addr = os.environ.get('MASTER_ADDR', 'localhost')
    is_ipv6 = len(root_addr.split(":")) >= 3
    if is_ipv6:
      modified = False
      if not root_addr.startswith("["):
        root_addr = "[" + root_addr
        modified = True
      if not root_addr.endswith("]"):
        root_addr = root_addr + "]"
        modified = True
      if modified:
        logger.warning(
            "IPv6 address detected for MASTER_ADDR and missing brackets added: {}"
            .format(root_addr))
    os.environ['NEURON_RT_ROOT_COMM_ID'] = '{}:{}'.format(root_addr, root_port)


def set_envvar_defaults():
  os.environ.setdefault('ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB', '50')


def configure_pjrt_environment():
  """
  Setting all necessary PJRT default environment variables.
  """
  from torch.distributed import is_torchelastic_launched

  # Set root communication address/port
  set_rt_root_comm_id()

  # Set env variables if we don't use GSPMD, using PJRT, and using torchrun
  if os.environ.get('XLA_USE_SPMD', '0') != '1' \
      and is_torchelastic_launched():
    # Env variables that only need to be set once
    # NEURON_PJRT_PROCESSES_NUM_DEVICES is a list of core counts and is too long for very large cluster,
    # so use NEURON_PJRT_WORLD_SIZE to pass world size and use core count of 1 per process in PJRT client.
    if 'NEURON_PJRT_PROCESSES_NUM_DEVICES' not in os.environ and 'NEURON_PJRT_WORLD_SIZE' not in os.environ:
      if 'WORLD_SIZE' not in os.environ:
        logger.warning(
            'WORLD_SIZE environment variable not set, defaulting to 1.')
      os.environ["NEURON_PJRT_WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
      if 'LOCAL_WORLD_SIZE' not in os.environ:
        logger.warning(
            'LOCAL_WORLD_SIZE environment variable not set, defaulting to 1.')
      os.environ['PJRT_LOCAL_PROCESS_COUNT'] = os.environ.get(
          'LOCAL_WORLD_SIZE', '1')

    # Env variables that need to be set once per process
    if not os.environ.get('NEURON_RT_VISIBLE_CORES', None):
      os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ.get('LOCAL_RANK', '0')
    else:
      local_rank = int(os.environ.get('LOCAL_RANK', '0'))
      local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
      remap_visible_cores(local_rank, local_world_size)

    if 'RANK' not in os.environ:
      logger.warning('RANK environment variable is not set, defaulting to 0.')
    os.environ['NEURON_PJRT_PROCESS_INDEX'] = os.environ.get('RANK', '0')
    if 'LOCAL_RANK' not in os.environ:
      logger.warning(
          'LOCAL RANK environment variable is not set, defaulting to 0.')
    os.environ['PJRT_LOCAL_PROCESS_RANK'] = os.environ.get('LOCAL_RANK', '0')


def num_local_processes() -> int:
  set_rt_root_comm_id()
  num_processes = int(os.environ.get("NEURONCORE_NUM_DEVICES", "1"))
  os.environ['NEURON_PJRT_PROCESSES_NUM_DEVICES'] = ','.join(
      ['1' for _ in range(num_processes)])
  return num_processes


# When torchrun is used, setting these environments causes the
# second instance in 2-node cluster to think it is node 0 instead of node 1.
# Need to skip these settings and let configure_pjrt_environment to
# set the distributed PJRT environment variables.
# If NEURONCORE_NUM_DEVICES is used, then go ahead and set the environments.
def initialize_env(local_rank, local_world_size):
  from torch.distributed import is_torchelastic_launched
  if not is_torchelastic_launched():
    os.environ["NEURON_PJRT_PROCESS_INDEX"] = str(local_rank)
    if not os.environ.get('NEURON_RT_VISIBLE_CORES', None):
      os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)
    else:
      remap_visible_cores(local_rank, local_world_size)


class NeuronPlugin(plugins.DevicePlugin):

  def library_path(self):
    from libneuronxla.libneuronpjrt_path import libneuronpjrt_path
    return os.environ.get("NEURON_LIBRARY_PATH", libneuronpjrt_path())

  def configure_multiprocess(self, local_rank, local_world_size):
    initialize_env(local_rank, local_world_size)

  def physical_chip_count(self):
    return num_local_processes()
