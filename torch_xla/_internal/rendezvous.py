import datetime
import logging
import threading
import os

import torch.distributed as dist
from torch_xla.distributed import xla_backend
from torch_xla import runtime as xr
from torch_xla._internal import pjrt
from torch_xla._internal import tpu, gpu
import torch_xla.utils.utils as xu

_store = None
_store_lock = threading.Lock()


def pjrt_rendezvous_handler(url: str,
                            timeout: datetime.timedelta = ...,
                            **kwargs):
  # Assume `xmp.spawn` has not been called when using torchrun
  print('xw32 pjrt_backend._pjrt_rendezvous_handler begins')
  if dist.is_torchelastic_launched():
    local_world_size = xu.getenv_as('LOCAL_WORLD_SIZE', int)
    local_rank = xu.getenv_as('LOCAL_RANK', int)
    print('xw32 pjrt_backend._pjrt_rendezvous_handler calling pjrt.initialize_multiprocess')

    # initialize dist server for GPU
    global_world_size = xu.getenv_as('WORLD_SIZE', int)
    global_rank = xu.getenv_as('RANK', int)
    print('xw32 pjrt_backend._pjrt_rendezvous_handler: global_world_size=', global_world_size, ', global_rank=', global_rank)
    if xr.device_type() == 'GPU' and global_rank == 0:
      gpu.initialize_distributed_runtime(global_world_size)
    # Need to create the dist server here.
    pjrt.initialize_multiprocess(local_rank, local_world_size)

  master_ip = xu.getenv_as('MASTER_ADDR', str)
  if not master_ip:
    master_ip = tpu.discover_master_worker_ip() if xr.device_type(
    ) == 'TPU' else 'localhost'

  master_port = xu.getenv_as('MASTER_PORT', int, 12355)
  world_size = xr.world_size()
  with _store_lock:
    global _store
    print('xw32 pjrt_backend._pjrt_rendezvous_handler creating dist.TCPStore')
    if not _store:
      if xu.getenv_as('TORCHELASTIC_USE_AGENT_STORE', str) == 'True':
        print('xw32 pjrt_backend._pjrt_rendezvous_handler TORCHELASTIC_USE_AGENT_STORE is True')
        attempt = xu.getenv_as('TORCHELASTIC_RESTART_COUNT', int, defval=0)
        tcp_store = dist.TCPStore(
            master_ip, master_port, xr.process_count(), is_master=False)
        _store = dist.PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
      else:
        print('xw32 pjrt_backend._pjrt_rendezvous_handler TORCHELASTIC_USE_AGENT_STORE is False')
        _store = dist.TCPStore(
            master_ip,
            master_port,
            xr.process_count(),
            is_master=xr.process_index() == 0)
      print('xw32 pjrt_backend._pjrt_rendezvous_handler dist.TCPStore has been created.')

  yield (_store, xr.global_ordinal(), world_size)
