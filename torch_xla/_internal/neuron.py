import os
import logging

from torch_xla.experimental import plugins
from torch_xla import runtime as xr

import sys
import torch.distributed as dist

from .neuron_utils import get_visible_cores_list, remap_visible_cores

logging.basicConfig()
logger = logging.getLogger(__name__)
# Singleton initializer to ensure that the initialization is only set once.
_initializer = None


def initialize():
  global _initializer
  if not _initializer:
    _initializer = Initializer()
  _initializer.reset()


# Set root communication address/port
def set_rt_root_comm_id():
  if os.environ.get('NEURON_RT_ROOT_COMM_ID', None) is None:
    if 'MASTER_ADDR' not in os.environ:
      logging.warning(
        "MASTER_ADDR environment variable is not set, defaulting to localhost"
      )
    root_port = 62182
    root_addr = os.environ.get('MASTER_ADDR', 'localhost')
    is_ipv6 = len(root_addr.split(':')) >= 3
    if is_ipv6:
      modified = False
      if not root_addr.startswith('['):
        root_addr = '[' + root_addr
        modified = True
      if not root_addr.endswith(']'):
        root_addr = root_addr + ']'
        modified = True
      if modified:
        logger.warning(
          "IPv6 address detected for MASTER_ADDR and missing brackets added: {}".format(
            root_addr
          )
        )
    os.environ['NEURON_RT_ROOT_COMM_ID'] = "{}:{}".format(root_addr, root_port)


def _set_envvar_defaults():
  os.environ.setdefault('ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB', '50')


class Initializer:
  """
  Initializer class that manages the initialization for torch. It cohesively
  guarantees that the environment is correctly configured for both SPMD and
  non-SPMD use cases. Note that in case SPMD is enabled, the initialization
  requires reconfiguring the environment, as this follows the default
  initialization.
  """

  # Whether the PJRT environment has already been configured.
  configured_pjrt_env = False
  # The previous state of the PJRT environment before the latest
  # configuration.
  previous_pjrt_env_vars = {}

  def __init__(self):
    import libneuronxla

    libneuronxla.configure_environment()
    _set_envvar_defaults()
    # Environment agnostic PJRT configurations that only need to be set once.
    self._initialize_pjrt_ranks()

  def reset(self):
    if self.configured_pjrt_env:
      self.__clear_previous_pjrt_env_vars()
    assert not (self.previous_pjrt_env_vars or self.configured_pjrt_env)
    self._configure_pjrt_environment()
    self.configured_pjrt_env = True

  def _initialize_pjrt_ranks(self):
    """
    Initialize the PJRT specific ranks for torch.
    """
    if 'RANK' not in os.environ:
      logger.warning("RANK environment variable is not set, defaulting to 0.")
    self.__set_envvar_defaulted_and_save('NEURON_PJRT_PROCESS_INDEX', 'RANK', '0')
    os.environ['NEURON_PJRT_PROCESS_INDEX'] = os.environ.get('RANK', '0')
    if 'LOCAL_RANK' not in os.environ:
      logger.warning(
        "LOCAL RANK environment variable is not set to 0, defaulting to 0."
      )
    self.__set_envvar_defaulted_and_save('PJRT_LOCAL_PROCESS_RANK', 'LOCAL_RANK', '0')

  def _configure_pjrt_environment(self):
    """
    Setting all necessary PJRT default environment variables. There are currently two schemes:
    - __configure_non_spmd_environment, for the non-SPMD setup.
    - __configure_spmd_environment, for the SPMD setup.
    """

    def __configure_non_spmd_environment():
      """
      Setting all necessary PJRT environment variables for non-SPMD::
          1) NEURON_PJRT_PROCESSES_NUM_DEVICES: `X,Y,Z` will denote X, Y and Z worker processes, each with
          one addressable device.
          2) NEURON_PJRT_WORLD_SIZE: This will denote the total number of worker processes, each with one
          addressable device. For instance, '8' will expand to '1,1,1,1,1,1,1,1'.
          3) NEURON_RT_VISIBLE_CORES: The specified visible cores are unwrapped and assigned to the
          corresponding local rank in order associated with its index.
          4) Default behavior:
          * NEURON_PJRT_WORLD_SIZE is overwritten to WORLD_SIZE, denoting the global number of participating
              devices.
          * PJRT_LOCAL_PROCESS_COUNT is overwritten to LOCAL_WORLD_SIZE, denoting the number of local
              participating processes.
      """
      # NEURON_PJRT_PROCESSES_NUM_DEVICES is a list of core counts and is too long for very large cluster,
      # so use NEURON_PJRT_WORLD_SIZE to pass world size and use core count of 1 per process in PJRT client.
      if (
        'NEURON_PJRT_PROCESSES_NUM_DEVICES' not in os.environ
        and 'NEURON_PJRT_WORLD_SIZE' not in os.environ
      ):
        if 'WORLD_SIZE' not in os.environ:
          logger.warning("WORLD_SIZE environment variable not set, defaulting to 1.")
        self.__set_envvar_defaulted_and_save(
          'NEURON_PJRT_WORLD_SIZE', 'WORLD_SIZE', '1'
        )
        if 'LOCAL_WORLD_SIZE' not in os.environ:
          logger.warning(
            "LOCAL_WORLD_SIZE environment variable not set, defaulting to 1."
          )
        self.__set_envvar_defaulted_and_save(
          'PJRT_LOCAL_PROCESS_COUNT', 'LOCAL_WORLD_SIZE', '1'
        )
      visible_cores = get_visible_cores_list()
      self.__set_envvar_defaulted_and_save(
        'NEURON_RT_VISIBLE_CORES',
        'LOCAL_RANK',
        '0' if not visible_cores else visible_cores,
      )

    def __configure_spmd_environment():
      """
      Setting all necessary PJRT environment variables for SPMD:
          1) NEURON_PJRT_PROCESSES_NUM_DEVICES: `X,Y,Z` will denote X, Y and Z addressable devices
          for the single worker process in the respective three node.
          2) Default behaviors
              * Single-node:
                  Use a single worker process that has all visible neuron cores:
                  * NEURON_RT_VISIBLE_CORES / NEURON_RT_VISIBLE_CORES if specified
                  * Otherwise, all available neuron cores in the instance.
              * Multi-node:
                  No default support, requires 1)
      """
      # In SPMD XRT, 'WORLD_SIZE' represents the global number of participant nodes.
      if 'WORLD_SIZE' not in os.environ:
        logger.warning("WORLD_SIZE environment variable not set, defaulting to 1.")
      if 'LOCAL_WORLD_SIZE' not in os.environ:
        logger.warning(
          "LOCAL_WORLD_SIZE environment variable not set, defaulting to 1."
        )
      self.__set_envvar_defaulted_and_save(
        'PJRT_LOCAL_PROCESS_COUNT', 'LOCAL_WORLD_SIZE', '1'
      )

      # 'NEURON_PJRT_PROCESSES_NUM_DEVICES' is required for multi-node support.
      assert (
        os.environ.get('WORLD_SIZE', '1') == '1'
        or 'NEURON_PJRT_PROCESSES_NUM_DEVICES' in os.environ
      ), (
        "NEURON_PJRT_PROCESSES_NUM_DEVICES environment variable not set. This is required to enable "
        "multi-node SPMD."
      )
      if 'NEURON_RT_VISIBLE_CORES' in os.environ:
        # In SPMD, we do not remap the visible cores based on the local work rank, but instead
        # just unwrap the visible cores if specified.
        self.__set_envvar_defaulted_and_save(
          'NEURON_RT_VISIBLE_CORES', None, get_visible_cores_list()
        )

    from torch.distributed import is_torchelastic_launched

    # If not using XRT, then do not set the environment variables. In this
    # case, the environment variables are initialized in the default
    # initializer with `initialize_env`.
    if not is_torchelastic_launched():
      return

    if xr.is_spmd():
      __configure_spmd_environment()
    else:
      __configure_non_spmd_environment()

  def __clear_previous_pjrt_env_vars(self):
    """
    Reset the environment variables for the PJRT environment to its former
    state.
    """
    assert self.configured_pjrt_env
    logger.warning("Reinitializing the PJRT environment.")
    if self.previous_pjrt_env_vars:
      # Reset the environment to a clean state
      for key, previous_val in self.previous_pjrt_env_vars.items():
        os.environ[key] = previous_val
    self.previous_pjrt_env_vars = {}
    self.configured_pjrt_env = False

  def __set_envvar_defaulted_and_save(self, key_to, key_from, default_value):
    """
    This is used to set a default value for an environment variable if it
    is not already set, and then save the original value of the environment
    variable to track its state in case we require re-initializing the
    environment.
    """
    if callable(default_value):
      default_value = default_value()
    value = os.environ.get(key_from, default_value) if key_from else default_value
    if key_to in os.environ and os.environ[key_to] != value:
      logger.debug(f"{key_to} environment variable is set, overriding to {value}.")
    os.environ[key_to] = value
    self.previous_pjrt_env_vars[key_to] = value


def num_local_processes() -> int:
  set_rt_root_comm_id()
  num_processes = int(os.environ.get('NEURONCORE_NUM_DEVICES', '1'))
  os.environ['NEURON_PJRT_PROCESSES_NUM_DEVICES'] = ','.join([
    '1' for _ in range(num_processes)
  ])
  return num_processes


# When torchrun is used, setting these environments causes the
# second instance in 2-node cluster to think it is node 0 instead of node 1.
# Need to skip these settings and let configure_pjrt_environment to
# set the distributed PJRT environment variables.
# If NEURONCORE_NUM_DEVICES is used, then go ahead and set the environments.
def initialize_env(local_rank, local_world_size):
  from torch.distributed import is_torchelastic_launched

  if not is_torchelastic_launched():
    os.environ['NEURON_PJRT_PROCESS_INDEX'] = str(local_rank)
    if not os.environ.get('NEURON_RT_VISIBLE_CORES', None):
      os.environ['NEURON_RT_VISIBLE_CORES'] = str(local_rank)
    else:
      remap_visible_cores(local_rank, local_world_size)


class NeuronPlugin(plugins.DevicePlugin):
  def library_path(self):
    from libneuronxla.libneuronpjrt_path import libneuronpjrt_path

    return os.environ.get('NEURON_LIBRARY_PATH', libneuronpjrt_path())

  def configure_multiprocess(self, local_rank, local_world_size):
    initialize_env(local_rank, local_world_size)

  def physical_chip_count(self):
    return num_local_processes()
