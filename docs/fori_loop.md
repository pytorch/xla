# `Fori_loop` | `While_loop` | `Scan` optimize memory utilization and compilation


### `fori_loop`

`fori_loop` replace pure python `for` loop, PyTorch/XLA would enable `torch_xla.experimental.fori_loop` to keep loop computation graph as rolled during compilation
like [`jax.lax.fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html), not like currently repeat computations by enumerating all execution steps
of each iteration. `fori_loop` help optimize memory utilization and compilation.

#### Usage:
```python
from torch_xla.experimental.fori_loop import fori_loop
res = fori_loop(lower, upper, body_fun, init)
```
- `lower`, `upper`: Loop boundaries.
- `body_fun`: User-defined loop-body-operations function.
- `init`: Initial value in the loop.

Currently, `fori_loop` support simple test, and can be explored on TPUs using [user guide](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#simple-example-with-fori_loop).

#### Implementation:
- **dynamic loop range**: [`fori_loop`](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#fori_loop) is implemented with [`while_loop`](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#while_loop)

  Like [`jax.lax.while_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html), PyTorch/XLA would support `while_loop` with native PyTorch and XLA 
  backend via `XLA::While`. Due to `while_loop` didn't support autograd, so it would be used for inference only.

- **not dynamic loop range**: [`fori_loop`](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#fori_loop) is implemented with [`scan`](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#wipscan)

  Like [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html), PyTorch/XLA would enable `scan` using `XLA::While` operator.
  This implementation would be very similar like `while_loop`. `scan` support autograd, and it could be used in both training and inference.

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

Currently, while_loop support simple test, and can be explored on TPUs using [user guide](https://github.com/pytorch/xla/blob/master/docs/fori_loop.md#simple-example-with-while_loop).

<br>

### `scan` (Work in Progress)
like [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html), PyTorch/XLA would enable `scan` for training and inference since it support autograd.

<br>

---

## Simple user guide
User could try these three simple test case to better compare difference between `pure python for loop` and `fori_loop` and `while_loop`, these three test case have similar logic: cumulative plus 1 for ten times:

### simple example with pure python for loop
```bash
# python
>>> import torch
>>> init = torch.tensor([0], dtype=torch.int32)
>>> one_value = torch.ones(1, dtype=torch.int32)
>>> 
>>> for i in range(10):
...   init = init + one_value
... 
>>> init
tensor([10], dtype=torch.int32)
```

### simple example with `while_loop`:
```bash
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.experimental.fori_loop
>>> from torch_xla.experimental.fori_loop import fori_loop
>>> from torch._higher_order_ops.while_loop import while_loop
>>> import torch_xla.core.xla_model as xm
>>> import torch_xla.core.xla_builder as xb
>>> 
>>> device = xm.xla_device()
>>> 
>>> def cond_fn(init, limit_value):
...   return limit_value[0] >= init[0]
... 
>>> def body_fn(init, limit_value):
...   one_value = torch.ones(1, dtype=torch.int32, device=device)
...   return (torch.add(init, one_value), limit_value.clone())
... 
>>> init = torch.tensor([0], dtype=torch.int32, device=device)
>>> limit_value = torch.tensor([10], dtype=torch.int32, device=device)
>>> res_, limit_value_ = while_loop(cond_fn, body_fn, (init, limit_value))
>>> res_
FunctionalTensor(lvl=0, value=\
tensor([11], device='xla:0', dtype=torch.int32))
```

### simple example with `fori_loop`:
```bash
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.experimental.fori_loop
>>> from torch_xla.experimental.fori_loop import fori_loop
>>> from torch._higher_order_ops.while_loop import while_loop
>>> import torch_xla.core.xla_model as xm
>>> import torch_xla.core.xla_builder as xb
>>> 
>>> device = xm.xla_device()
>>> 
>>> lower = torch.tensor([2], dtype=torch.int32, device=device)
>>> upper = torch.tensor([52], dtype=torch.int32, device=device)
>>> plus_value = torch.tensor([1], dtype=torch.int32, device=device)
>>> init_val = torch.tensor([1], dtype=torch.int32, device=device)
>>> 
>>> def body_fun(*argus):
...   plus_value, init_val = argus
...   return plus_value, torch.add(plus_value, init_val)
... 
>>> _, _, _, res_ = fori_loop(upper, lower, body_fun, plus_value, init_val)
>>> res_
tensor([51], device='xla:0', dtype=torch.int32)
```

PyTorch/XLA would include `while_loop` support in 2.4 with test case, support for `fori_loop` and `scan` would be added after 2.4. For `fori_loop` and `while_loop`, currently we only should force define `body_fn` with same `input` and `output(return args)` shape
