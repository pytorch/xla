import datetime
import logging
import threading

import torch.distributed as dist
from torch_xla.distributed import xla_backend
from torch_xla import runtime as xr
from torch_xla._internal import pjrt
from torch_xla._internal import tpu
import torch_xla.utils.utils as xu

_store = None
_store_lock = threading.Lock()


def _pjrt_rendezvous_handler(url: str,
                             timeout: datetime.timedelta = ...,
                             **kwargs):
  # Assume `xmp.spawn` has not been called when using torchrun
  if dist.is_torchelastic_launched():
    local_world_size = xu.getenv_as('LOCAL_WORLD_SIZE', int)
    local_rank = xu.getenv_as('LOCAL_RANK', int)
    pjrt.initialize_multiprocess(local_rank, local_world_size)

  master_ip = xu.getenv_as('MASTER_ADDR', str)
  if not master_ip:
    master_ip = tpu.discover_master_worker_ip() if xr.device_type(
    ) == 'TPU' else 'localhost'

  master_port = xu.getenv_as('MASTER_PORT', int, 12355)
  world_size = xr.world_size()
  with _store_lock:
    global _store
    if not _store:
      if xu.getenv_as('TORCHELASTIC_USE_AGENT_STORE', str) == 'True':
        attempt = xu.getenv_as('TORCHELASTIC_RESTART_COUNT', int, defval=0)
        tcp_store = dist.TCPStore(
            master_ip, master_port, xr.process_count(), is_master=False)
        _store = dist.PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
      else:
        _store = dist.TCPStore(
            master_ip,
            master_port,
            xr.process_count(),
            is_master=xr.process_index() == 0)

  yield (_store, xr.global_ordinal(), world_size)


if tpu.num_available_chips() > 0 and tpu.version() <= 3:
  from torch.testing._internal.distributed import multi_threaded_pg
  logging.warning('Patching torch.distributed state to support multithreading.')
  logging.warning('torch.distributed support on TPU v2 and v3 is experimental '
                  'and does not support torchrun.')
  multi_threaded_pg._install_threaded_pg()

dist.register_rendezvous_handler('pjrt', _pjrt_rendezvous_handler)
