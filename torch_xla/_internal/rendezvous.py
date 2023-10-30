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


def pjrt_rendezvous_handler(url: str,
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

  # In SPMD, the world size and rank are determined by the process count and
  # index, while in multiprocess they are based on the device count and ordinal.
  world_size = xr.process_count() if xr.is_spmd() else xr.world_size()
  rank = xr.process_index() if xr.is_spmd() else xr.global_ordinal()
  yield (_store, rank, world_size)
