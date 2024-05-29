import sys

assert len(sys.argv) in (2, 3)
devkind = sys.argv[1]
use_xla2 = False
if len(sys.argv) == 3 and sys.argv[2].lower() == 'true':
  use_xla2 = True

import os
os.environ["PJRT_DEVICE"] = devkind

if not use_xla2:
  import torch_xla.core.xla_model as xm
  devlist = xm.get_xla_supported_devices(devkind=devkind)
else:
  # torch_xla2 needs jax to detect the device
  os.environ["JAX_PLATFORMS"] = devkind.lower(
  )  # JAX_PLATFORMS only accepts lower case
  assert devkind.lower() in ('cpu', 'gpu', 'tpu')
  import jax
  devlist = jax.devices(devkind.lower())

if not devlist:
  sys.exit(1)
