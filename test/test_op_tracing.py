import torch
import torch_xla
#torch_xla._XLAC._ltc_init_ts_backend() #TOREMOVE
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as metrics
import traceback

unique_lines = set()

def print_stacktrace():
    global unique_lines
    traceback.print_stack()
    stack = traceback.extract_stack()
    stack = traceback.format_list(stack)
    if (len(stack) > 1):
        second_frame = str(stack[-2]).replace("/usr/local/google/home/miladmo/anaconda3/envs/pytorch1/bin/ipython", "")
        unique_lines.add(second_frame)

torch_xla._XLAC._set_custom_printer(print_stacktrace)

d = torch.rand(3, 3, device="cuda", requires_grad=True)
a = d.detach().clone().to(device="xla").requires_grad_(True)
c = torch.rand(1, 1, device="xla")
b = a.view(c.size(1), 9)
b.sum().backward()
print(a.grad.to(device="cpu").sum())
print("done!")

print(f"unique lines {len(unique_lines)} {id(unique_lines)}")
for l in unique_lines:
    print(l)
print("done")
