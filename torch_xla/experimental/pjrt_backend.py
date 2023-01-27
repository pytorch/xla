import datetime
import threading

import torch.distributed as dist
from torch.testing._internal.distributed import multi_threaded_pg
from torch_xla.distributed import xla_backend
from torch_xla.experimental import pjrt, tpu
import torch_xla.utils.utils as xu

_store = None
_store_lock = threading.Lock()


def _pjrt_rendezvous_handler(url: str,
                             timeout: datetime.timedelta = ...,
                             **kwargs):
  master_ip = xu.getenv_as('MASTER_ADDR', str)
  if not master_ip:
    master_ip = tpu.discover_master_worker_ip() if pjrt.device_type(
    ) == 'TPU' else 'localhost'

  master_port = xu.getenv_as('MASTER_PORT', int, 12355)
  with _store_lock:
    global _store
    if not _store:
      _store = dist.TCPStore(
          master_ip,
          master_port,
          pjrt.process_count(),
          is_master=pjrt.process_index() == 0)

  yield (_store, pjrt.global_ordinal(), pjrt.world_size())


multi_threaded_pg._install_threaded_pg()

dist.register_rendezvous_handler('pjrt', _pjrt_rendezvous_handler)
