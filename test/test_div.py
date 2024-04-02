import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

dev=xm.xla_device()
t1 = torch.tensor(3.0, device=dev, dtype=torch.bfloat16) # calls XLANativeFunctions::_to_copy, XLANativeFunctions::_copy_from, XLANativeFunctions::life_fresh
t2 = torch.tensor(1.0, device=dev, dtype=torch.float64)

# import pdb; pdb.set_trace()
#print(torch_xla._XLAC._get_xla_tensors_text([t1]))
# IR {
#   %0 = f32[] xla::device_data(), xla_shape=f32[], ROOT=0
# }

#print(torch_xla._XLAC._get_xla_tensors_hlo([t1]))
# (Pdb) n
# HloModule IrToHlo.3, entry_computation_layout={(f32[])->(f32[])}
# 
# ENTRY %IrToHlo.3 (p0.1: f32[]) -> (f32[]) {
#   %p0.1 = f32[] parameter(0)
#   ROOT %tuple.2 = (f32[]) tuple(f32[] %p0.1)
# }

t3 = torch.div(t1, t2)
print(torch_xla._XLAC._get_xla_tensors_text([t3]))
print(t3.dtype)
# IR {
#   %0 = f32[] prim::Constant(), xla_shape=f32[]
#   %1 = f32[] xla::device_data(), xla_shape=f32[]
#   %2 = f32[] aten::div(%1, %0), xla_shape=f32[], ROOT=0
# }

print(torch_xla._XLAC._get_xla_tensors_hlo([t3]))
# HloModule IrToHlo.5, entry_computation_layout={(f32[])->(f32[])}
# 
# ENTRY %IrToHlo.5 (p0.2: f32[]) -> (f32[]) {
#   %p0.2 = f32[] parameter(0)
#   %constant.1 = f32[] constant(1)
#   %divide.3 = f32[] divide(f32[] %p0.2, f32[] %constant.1)
#   ROOT %tuple.4 = (f32[]) tuple(f32[] %divide.3)
# }

xm.mark_step()
print(t3.dtype)
#print(torch_xla._XLAC._get_xla_tensors_text([t1]))
# IR {
#   %0 = f32[] xla::device_data(), xla_shape=f32[], ROOT=0
# }

#print(torch_xla._XLAC._get_xla_tensors_hlo([t1]))
# HloModule IrToHlo.3, entry_computation_layout={(f32[])->(f32[])}
# 
# ENTRY %IrToHlo.3 (p0.1: f32[]) -> (f32[]) {
#   %p0.1 = f32[] parameter(0)
#   ROOT %tuple.2 = (f32[]) tuple(f32[] %p0.1)
# }

#print(torch_xla._XLAC._get_xla_tensors_text([t3]))
# IR {
#   %0 = f32[] xla::device_data(), xla_shape=f32[], ROOT=0
# }

#print(torch_xla._XLAC._get_xla_tensors_hlo([t3]))
# HloModule IrToHlo.3, entry_computation_layout={(f32[])->(f32[])}
# 
# ENTRY %IrToHlo.3 (p0.1: f32[]) -> (f32[]) {
#   %p0.1 = f32[] parameter(0)
#   ROOT %tuple.2 = (f32[]) tuple(f32[] %p0.1)
# }
# print(met.metrics_report())

