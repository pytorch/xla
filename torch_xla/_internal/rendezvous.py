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


class DummyStore(dist.Store):

  def __init__(self, *args, **kwargs):
    super().__init__()


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
  world_size = xr.world_size()
  with _store_lock:
    global _store
    if not _store:
      # Create DummyStore when user skips store based barrier by setting TORCH_DIST_INIT_BARRIER=0
      # and enables XLA_USE_DUMMY_STORE=1. It's safe to do so because store created by _pjrt_rendezvous_handler
      # is only used as a barrier in process groups. If store is needed, user can set XLA_USE_DUMMY_STORE=0 to
      # use TCPStore.
      if xu.getenv_as('TORCH_DIST_INIT_BARRIER', int, 1) == 0 and xu.getenv_as(
          'XLA_USE_DUMMY_STORE', int, 0) == 1:
        _store = DummyStore()
      elif xu.getenv_as('TORCHELASTIC_USE_AGENT_STORE', str) == 'True':
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
