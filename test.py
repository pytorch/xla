import torch, torch_xla
from torch_xla import runtime as xr

xr.device_type()
xr.local_process_count()
xr.global_device_count()
xr.world_size()
xr.local_device_count()
xr.addressable_device_count()
xr.global_ordinal()
xr.local_ordinal()
xr.process_index()
xr.process_count()
xr.global_runtime_device_attributes()
xr.global_runtime_device_count()
xr.addressable_runtime_device_count()
xr.get_master_ip()
