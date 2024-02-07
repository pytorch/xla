import sys
assert len(sys.argv) == 2
devkind = sys.argv[1]

import os
os.environ["PJRT_DEVICE"] = devkind

import torch_xla.core.xla_model as xm
devlist = xm.get_xla_supported_devices(devkind=devkind)

if devlist is None or len(devlist) == 0:
  sys.exit(1)
