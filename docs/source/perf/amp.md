# Automatic Mixed Precision

Pytorch/XLA's AMP extends [Pytorch's AMP
package](https://pytorch.org/docs/stable/amp.html) with support for
automatic mixed precision on `XLA:GPU` and `XLA:TPU` devices. AMP is
used to accelerate training and inference by executing certain
operations in `float32` and other operations in a lower precision
datatype (`float16` or `bfloat16` depending on hardware support). This
document describes how to use AMP on XLA devices and best practices.

## AMP for XLA:TPU

AMP on TPUs automatically casts operations to run in either `float32` or
`bfloat16` because TPUs natively support bfloat16. A simple TPU AMP
example is below:

``` python
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

`autocast(xm.xla_device())` aliases `torch.autocast('xla')` when the XLA
Device is a TPU. Alternatively, if a script is only used with TPUs, then
`torch.autocast('xla', dtype=torch.bfloat16)` can be directly used.

Please file an issue or submit a pull request if there is an operator
that should be autocasted that is not included.

### AMP for XLA:TPU Best Practices

1.  `autocast` should wrap only the forward pass(es) and loss
    computation(s) of the network. Backward ops run in the same type
    that autocast used for the corresponding forward ops.
2.  Since TPU's use bfloat16 mixed precision, gradient scaling is not
    necessary.
3.  Pytorch/XLA provides modified version of
    [optimizers](https://github.com/pytorch/xla/tree/master/torch_xla/amp/syncfree)
    that avoid the additional sync between device and host.

### Supported Operators

AMP on TPUs operates like Pytorch's AMP. Rules for how autocasting is
applied is summarized below:

Only out-of-place ops and Tensor methods are eligible to be autocasted.
In-place variants and calls that explicitly supply an out=... Tensor are
allowed in autocast-enabled regions, but won't go through autocasting.
For example, in an autocast-enabled region a.addmm(b, c) can autocast,
but a.addmm\_(b, c) and a.addmm(b, c, out=d) cannot. For best
performance and stability, prefer out-of-place ops in autocast-enabled
regions.

Ops that run in float64 or non-floating-point dtypes are not eligible,
and will run in these types whether or not autocast is enabled.
Additionally, Ops called with an explicit dtype=... argument are not
eligible, and will produce output that respects the dtype argument.

Ops not listed below do not go through autocasting. They run in the type
defined by their inputs. Autocasting may still change the type in which
unlisted ops run if they're downstream from autocasted ops.

**Ops that autocast to `bfloat16`:**

`__matmul__`, `addbmm`, `addmm`, `addmv`, `addr`, `baddbmm`,`bmm`,
`conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`,
`conv_transpose3d`, `linear`, `matmul`, `mm`, `relu`, `prelu`,
`max_pool2d`

**Ops that autocast to `float32`:**

`batch_norm`, `log_softmax`, `binary_cross_entropy`,
`binary_cross_entropy_with_logits`, `prod`, `cdist`, `trace`, `chloesky`
,`inverse`, `reflection_pad`, `replication_pad`, `mse_loss`,
`cosine_embbeding_loss`, `nll_loss`, `multilabel_margin_loss`, `qr`,
`svd`, `triangular_solve`, `linalg_svd`, `linalg_inv_ex`

**Ops that autocast to widest input type:**

`stack`, `cat`, `index_copy`

## AMP for XLA:GPU

AMP on XLA:GPU devices reuse Pytorch's AMP rules. See [Pytorch's AMP
documentation](https://pytorch.org/docs/stable/amp.html) for CUDA
specific behavior. A simple CUDA AMP example is below:

``` python
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
    xm.all_reduce('sum', gradients, scale=1.0 / xr.world_size())
    scaler.step(optimizer)
    scaler.update()
```

`autocast(xm.xla_device())` aliases `torch.cuda.amp.autocast()` when the
XLA Device is a CUDA device (XLA:GPU). Alternatively, if a script is
only used with CUDA devices, then `torch.cuda.amp.autocast` can be
directly used, but requires `torch` is compiled with `cuda` support for
datatype of `torch.bfloat16`. We recommend using
`autocast(xm.xla_device())` on XLA:GPU as it does not require
`torch.cuda` support for any datatypes, including `torch.bfloat16`.

### AMP for XLA:GPU Best Practices

1.  `autocast` should wrap only the forward pass(es) and loss
    computation(s) of the network. Backward ops run in the same type
    that autocast used for the corresponding forward ops.
2.  Do not set `XLA_USE_F16` flag when using AMP on Cuda devices. This
    will override the per-operator precision settings provided by AMP
    and cause all operators to execute in float16.
3.  Use gradient scaling to prevent float16 gradients from underflowing.
4.  Pytorch/XLA provides modified version of
    [optimizers](https://github.com/pytorch/xla/tree/master/torch_xla/amp/syncfree)
    that avoid the additional sync between device and host.

## Examples

Our [mnist training script](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py)
and [imagenet training script](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_amp.py)
demonstrate how AMP is used on both TPUs and GPUs.
