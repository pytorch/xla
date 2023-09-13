import logging

import torch.distributed as dist
from torch_xla.distributed import xla_backend
from torch_xla._internal import rendezvous
from torch_xla._internal import tpu

if tpu.num_available_chips() > 0 and tpu.version() <= 3:
  from torch.testing._internal.distributed import multi_threaded_pg
  logging.warning('Patching torch.distributed state to support multithreading.')
  logging.warning('torch.distributed support on TPU v2 and v3 is experimental '
                  'and does not support torchrun.')
  multi_threaded_pg._install_threaded_pg()

dist.register_rendezvous_handler('pjrt', rendezvous.pjrt_rendezvous_handler)
