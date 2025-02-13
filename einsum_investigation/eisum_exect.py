import torch
import torch_xla.runtime

x = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3, requires_grad=True)

print("T1")
with torch.enable_grad():
  with torch_xla.runtime.xla_device():
    out_einsum = torch.einsum('ab,bc->ab', x.to('xla').requires_grad_(), y.to('xla').requires_grad_())


print("T2")
with torch.enable_grad():
  with torch_xla.runtime.xla_device():
    out_einsum = torch.einsum('...a,ab->...b', x.to('xla').requires_grad_(), y.to('xla').requires_grad_())

print("T3")
with torch.enable_grad():
  with torch_xla.runtime.xla_device():
    out_einsum = torch.einsum('...b,ab->...a', x.to('xla').requires_grad_(), y.to('xla').requires_grad_())

# TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE="aten_xla_type=1" python einsum_investigation/eisum_exect.py
# TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MAX_LOG_LEVEL=5 python einsum_investigation/eisum_exect.py
# TF_CPP_MIN_LOG_LEVEL=5 TF_CPP_MAX_LOG_LEVEL=0 python einsum_investigation/eisum_exect.py
