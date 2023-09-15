# Dynamic shape

Dynamic shape refers to the variable nature of a tensor shape where its shape depends on the value of another upstream tensor. For example:
```
>>> import torch, torch_xla
>>> in_tensor  = torch.randint(low=0, high=2, size=(5,5), device='xla:0')
>>> out_tensor = torch.nonzero(in_tensor)
```
the shape of `out_tensor` depends on the value of `in_tensor` and the shape is bounded by the shape of `in_tensor`. In other words, 
```
>>> print(out_tensor.shape)
torch.Size([<=25, 2])
```
You can see the first dimension depends on the value of `in_tensor` and its maximum value is 25. We call the first dimension as dynamic dimension. The second dimension does not depend on any upstream tensors so we call it static dimension.

Dynamic shape can be further categorized into bounded dynamic shape and unbounded dynamic shape.
- bounded dynamic shape: refers to a shape whose dynamic dimensions are bounded by static values. It works for accelerators that require static memory allocation (e.g. TPU).
- unbounded dynamic shape: refers to shape whose dynamic dimensions can be infinitely large. It works for accelerators that don’t require static memory allocation (e.g. GPU).

Currently, only bounded dynamic shape is supported.

## Bounded dynamic shape

