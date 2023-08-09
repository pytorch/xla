import datetime
import os
import threading

import torch.distributed as dist
from torch.testing._internal.distributed import multi_threaded_pg
from torch_xla.distributed import xla_backend
from torch_xla import runtime as xr
from torch_xla._internal import tpu
import torch_xla.utils.utils as xu

_store = None
_store_lock = threading.Lock()


def _pjrt_rendezvous_handler(url: str,
                             timeout: datetime.timedelta = ...,
                             **kwargs):
  master_ip = xu.getenv_as('MASTER_ADDR', str)
  if not master_ip:
    master_ip = tpu.discover_master_worker_ip() if xr.device_type(
    ) == 'TPU' else 'localhost'
  # Workaround for Neuron to prevent socket in use error with torchrun
  # MASTER_PORT is set by torchrun and will clash with the port here
  if dist.is_torchelastic_launched() and pjrt.device_type() == 'NEURON':
    master_port = xu.getenv_as('NEURON_PJRT_MASTER_PORT', int, 12355)
    # In the case of torchrun pjrt.world_size() is not yet initialized
    # until after the process group is initialized so we use the
    # world size variable from torchrun
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
  else:
    master_port = xu.getenv_as('MASTER_PORT', int, 12355)
    world_size = xr.world_size()
  with _store_lock:
    global _store
    if not _store:
      _store = dist.TCPStore(
          master_ip,
          master_port,
          xr.process_count(),
          is_master=xr.process_index() == 0)

  yield (_store, xr.global_ordinal(), world_size)


multi_threaded_pg._install_threaded_pg()

dist.register_rendezvous_handler('pjrt', _pjrt_rendezvous_handler)
