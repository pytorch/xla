# Fori_loop
`fori_loop` is a replacement of pure python for loop, like [`jax.lax.fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html):
- if loop range is dynamic, its deeper implementation would be [`jax.lax.while_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html),
- if loop range is not dynamic, its deeper implementation would be [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html),

In 2.3, PyTorch/XLA enable `while_loop` with some simple case, and

# while_loop
`while_loop` is a replacement of pure python while loop, PyTorch has supported `while_loop` in
[code](https://github.com/pytorch/pytorch/blob/ca6a0e1348ba7dcade1833d983b1b4ca12a5c1e1/torch/_higher_order_ops/while_loop.py#L69). 
PyTorch/XLA want to support `while_loop` with the native PyTorch and the XLA backend: XLA::While.

User could use `while_loop` like this:
```python
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
res = while_loop(/*user-defined*/cond_fn, /*user-defined*/body_fn, /*tuple or list*/init)
```
current while_loop only support simple test like [link](https://github.com/pytorch/xla/blob/r2.3/test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py), and user could try [simple user guide](https://github.com/pytorch/xla/blob/ManfeiBai-patch-81/docs/fori_loop.md#simple-example-with-while_loop) with `while_loop` on TPU too.


# [WIP]scan
not included in 2.3, please touch more in [nightly version](https://github.com/pytorch/xla/blob/r2.3/torch_xla/experimental/fori_loop.py)


# Simple user guide
User could try these simple test cases to better compare difference between `pure python for loop` and `while_loop`, these three test case have similar logic: cumulative plus 1 for ten times:

### simple example with pure python while loop
```bash
# python
>>> import torch
>>> 
>>> def cond_fn(init, limit_value):
...   return limit_value[0] > init[0]
... 
>>> def body_fn(init, limit_value):
...   one_value = torch.ones(1, dtype=torch.int32)
...   return (torch.add(init, one_value), limit_value.clone())
... 
>>> init = torch.tensor([0], dtype=torch.int32)
>>> limit_value = torch.tensor([10], dtype=torch.int32)
>>> 
>>> while cond_fn(init, limit_value):
...   init, _ = body_fn(init, limit_value)
... 
>>> init
tensor([10], dtype=torch.int32)
>>>
```

### simple example with `while_loop`:
```bash
# PJRT_DEVICE=TPU python
>>> import torch
>>> import torch_xla
>>> import torch_xla.experimental.fori_loop
>>> from torch._higher_order_ops.while_loop import while_loop
>>> import torch_xla.core.xla_model as xm
>>> import torch_xla.core.xla_builder as xb
>>> 
>>> device = xm.xla_device()
>>> 
>>> def cond_fn(init, limit_value):
...   return limit_value[0] > init[0]
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
tensor([10], device='xla:0', dtype=torch.int32))
>>>
```

For more example and detailed user guide, please read [this test file](https://github.com/pytorch/xla/blob/r2.3/test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py).
