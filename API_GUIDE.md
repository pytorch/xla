# Performance guideline for running on TPUs

## XLA tensors

PyTorch / XLA adds a new device, similarly to CPU and GPU devices. The following snippet creates an XLA tensor filled with random values, then prints the device and the contents of the tensor:

```
import torch
import torch_xla
import torch_xla_py.xla_model as xm

x = torch.randn(4, 2, device=xm.xla_device())
print(x.device)
print(x)
```

The XLA device is not a physical device but instead stands in for either a Cloud TPU or CPU.

The [XLA readme](https://github.com/pytorch/xla/blob/master/README.md) describes all the options available to run on TPU or CPU.

## Running a model

To run a model, use the following API:

```
import torch_xla_py.xla_model as xm
import torch_xla_py.data_parallel as dp

devices = xm.get_xla_supported_devices()
model_parallel = dp.DataParallel(MNIST, device_ids=devices)
                                                                                                                                                                                 
def train_loop_fn(model, loader, device, context):
  loss_fn = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  model.train()
  for _, (data, target) in loader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)

for epoch in range(1, num_epochs + 1):
  model_parallel(train_loop_fn, train_loader)
```

The same multi-core API can be used to run on a single core as well by setting the device_ids argument to the selected core. Passing `[]` as `device_ids` causes the model to run using the PyTorch native CPU support.

Note the `xm.optimizer_step(optimizer)` line which replaces the usual `optimizer.step()`. This is required because of the way XLA tensors work: operations are not executed immediately, but rather added to a graph of pending operations which is only executed when its results are required. Using `xm.optimizer_step(optimizer)` acts as an execution barrier which forces the evaluation of the graph accumulated for a single step. Without this barrier, the graph would only be evaluated when evaluating the accuracy of the model, which is only done at the end of an epoch, for this example. Even for small models, the accumulated graph would be too big to evaluate at the end of an entire epoch.

Check the [full example](https://github.com/pytorch/xla/blob/master/test/test_train_mnist.py) showing how to train MNIST on TPU.

## Performance caveats

PyTorch / XLA behaves semantically like regular PyTorch and XLA tensors, implementing the full tensor interface. However, constraints in XLA and hardware, and the lazy evaluation model mean some patterns must be avoided:

1.  Tensor shapes should be the same between iterations, or a low number of shape variations should be used. PyTorch / XLA automatically recompiles the graph every time new shapes are encountered. This means that, if the shapes don’t stabilize during training, more time will be spent compiling than running the model. Pad tensors to fixed sizes when possible. Direct or indirect uses of `nonzero` introduce dynamic shapes; for example, masked indexing `base[index]` where `index` is a mask tensor.
2.  Certain operations don’t have native translations to XLA and therefore require transfer to the CPU memory, evaluation on CPU, and transfer of the result back to the XLA device. This is automatically handled by PyTorch / XLA, but doing too many such operations during the training step can lead to significant slowdowns. The `item()` operation is one such example and it is used in [clip_grad_norm_](https://github.com/pytorch/pytorch/blob/de19eeee99a2a282fc441f637b23d8e50c75ecd1/torch/nn/utils/clip_grad.py#L33). Below is an alternative implementation which avoids the need for `item()`:

    ```
    ...
    else:
      device = parameters[0].device
      total_norm = torch.zeros([], device=device if parameters else None)
      for p in parameters:
        param_norm = p.grad.data.norm(norm_type) ** norm_type
        total_norm.add_(param_norm)
      total_norm = (total_norm ** (1. / norm_type))
    clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
    for p in parameters:
      p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
    ```


3.  Loops with a different number of iterations between steps are subject to similar observations as tensor shapes. PyTorch / XLA automatically handles them, but they are seen as different execution graphs and require recompilations.

    `print(torch_xla._XLAC._xla_metrics_report())` can be used to print metrics at the end of each step to collect information regarding the number of compilations and operators that are part of the model but don’t have native XLA implementations. The `XLA_METRICS_FILE=1` environment setting can also be used to export per step metrics to a file.

4. Sometimes model writers, when knowing that a PyTorch tensor is a scalar, they trigger `tensor.item()` (or equivalent PyTorch APIs which results to the same effects) calls, and they perform operations in Python scalar context, when similar operations can be performed using Pytorch tensor APIs. Following the latter approach will likely result in those operations behind fully fused within an XLA graph, without the need of issuing separate TPU computations.

   This can dramatically improve performance of the model, up to an N factor, where N is the number of `tensor.item()` calls per step.

