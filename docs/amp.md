# AMP (Automatic Mixed Precision) with Pytorch/XLA

Pytorch/XLA's AMP extends [Pytorch's AMP package](https://pytorch.org/docs/stable/amp.html) with support for automatic mixed precision on XLA:GPU and XLA:TPU devices. This document describes how to use AMP on XLA devices and best practices.

TODO: talk about how we follow PYtorch's rules and blah blah.

## AMP for XLA:TPU
TPUs natively support [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and float32 datatypes. 

```
# Creates model and optimizer in default precision
model = Net().to(xm.xla_device())
# Pytorch/XLA provides sync-free optimizers for improved performance
optimizer = syncfree.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass
    with autocast(xm.xla_device()):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    xm.optimizer_step.(optimizer)
```

### Best Practices
1. Do not set `XLA_USE_BF16` flag when using AMP on TPUs. This will override the per-operator precision settings provided by AMP and cause all operators to execute in bfloat16.
2. Since TPU's use bfloat16 mixed precision, gradient scaling is not required. 
TODO: What is the error message?
3. Pytorch/XLA provides modified version of [optimizers](https://github.com/pytorch/xla/tree/master/torch_xla/amp/syncfree) that avoid the additional sync between device and host. 

## AMP for XLA:GPU

```
# Creates model and optimizer in default precision
model = Net().to(xm.xla_device())
# Pytorch/XLA provides sync-free optimizers for improved performance
optimizer = syncfree.SGD(model.parameters(), ...)
scaler = GradScaler()

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass
    with autocast(xm.xla_device()):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward pass
    scaler.scale(loss).backward()
    gradients = xm._fetch_gradients(optimizer)
    xm.all_reduce('sum', gradients, scale=1.0 / xm.xrt_world_size())
    scaler.step(optimizer)
    scaler.update()
```

## Examples
Our [mnist training script](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py) and [imagenet training script](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_amp.py) demonstrate how AMP is used on both TPUs and GPUs. 

