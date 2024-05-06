import time
start_time = time.time()

import torch
import torch_xla
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.debug.profiler as xp

# server = xp.start_server(9012)

# xp.trace_detached(
#     f'localhost:9012',
#     '/root/profiles/',
#     duration_ms=2000)

device = xm.xla_device()

def cond_fn(init, limit_value):
    return limit_value[0] >= init[0]

def body_fn(init, limit_value):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (torch.add(init, one_value), limit_value.clone())

init = torch.tensor([0], dtype=torch.int32, device=device)
limit_value = torch.tensor([1000], dtype=torch.int32, device=device)
res = while_loop(cond_fn, body_fn, (init, limit_value))
print("res: ", res)

print("--- %s seconds ---" % (time.time() - start_time))
