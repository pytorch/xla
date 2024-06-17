# `While_loop` optimize memory utilization and compilation

<br>

### `while_loop`
`while_loop` replace pure python `while` loop, PyTorch supported `while_loop` by
[torch._higher_order_ops.while_loop](https://github.com/pytorch/pytorch/blob/62311257adb902d6a4ea98809c88895af1dbbf2b/torch/_higher_order_ops/while_loop.py#L66). 
PyTorch/XLA provide experimental XLA backend support for `torch._higher_order_ops.while_loop` via `XLA::While`.

#### Usage:
```python
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
result = while_loop(cond_fn, body_fn, init)
```
- `cond_fn`: User-defined condition function.
- `body_fn`: User-defined loop body function.
- `init`: Initial values (tuple or list).

#### simple example with `while_loop`:
```bash
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.experimental.fori_loop
>>> from torch._higher_order_ops.while_loop import while_loop
>>> import torch_xla.core.xla_model as xm
>>> 
>>> device = xm.xla_device()
>>> 
>>> def cond_fn(iteri, x):
...   return iteri > 0
... 
>>> def body_fn(iteri, x):
...   return iteri - 1, torch.add(x, 1)
... 
>>> init_val = torch.tensor(3, device=device)
>>> iteri = torch.tensor(10, device=device)
>>> _, res = while_loop(cond_fn, body_fn, (iteri, init_val))
>>> res
FunctionalTensor(lvl=0, value=\
tensor(13, device='xla:0'))
```

<br>

## Control group test case
For better compare difference between `pure python while loop` and `while_loop`, there is one test case called pure python `while` loop with similar logic: cumulative plus 1 for ten times:

### Control group example with pure python `while` loop
```bash
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.core.xla_model as xm
>>> 
>>> device = xm.xla_device()
>>> 
>>> init_val = torch.tensor(1, device=device)
>>> iteri = torch.tensor(50, device=device)
>>> 
>>> while iteri > 0:
...   init_val = init_val + 1
...   iteri -= 1
... 
>>> init_val
tensor(51, device='xla:0')
```



PyTorch/XLA would include `while_loop` support in 2.4 with test case, support for `fori_loop` would be added after 2.4. For `while_loop`, currently we only should force define `body_fn` with same `input` and `output(return args)` shape
