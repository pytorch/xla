import torch, torch_xla
from torch_xla import runtime as xr

print('xr.device_type()', xr.device_type())
print('xr.local_process_count()', xr.local_process_count())
print('xr.global_device_count()', xr.global_device_count())
print('xr.world_size()', xr.world_size())
print('xr.local_device_count()', xr.local_device_count())
print('xr.addressable_device_count()', xr.addressable_device_count())
print('xr.global_ordinal()', xr.global_ordinal())
print('xr.local_ordinal()', xr.local_ordinal())
print('xr.process_index()', xr.process_index())
print('xr.process_count()', xr.process_count())
print('xr.global_runtime_device_attributes()', xr.global_runtime_device_attributes())
print('xr.global_runtime_device_count()', xr.global_runtime_device_count())
print('xr.addressable_runtime_device_count()', xr.addressable_runtime_device_count())
print('xr.get_master_ip()', xr.get_master_ip())
