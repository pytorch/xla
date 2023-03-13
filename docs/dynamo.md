## TorchDynamo(torch.compile) integration in PyTorch XLA

[Torchdynamo](https://pytorch.org/tutorials/intermediate/dynamo_tutorial.html) is a Python-level JIT compiler designed to make unmodified PyTorch programs faster. It provides a clean API for compiler backends to hook in and its biggest feature is to dynamically modify Python bytecode right before it is executed. In the pytorch/xla 2.0 release, PyTorch/XLA provided an experimental backend for the TorchDynamo for both inference and training.

The way that XLA bridge works is that Dynamo will provide a TorchFX graph when it recognizes a model pattern and PyTorch/XLA will use existing Lazy Tensor technology to compile the FX graph and return the compiled function.

### Inference
Here is a small code example of running resnet18 with `torch.compile`

```python
import torch
imprt torchvision
import torch_xla.core.xla_model as xm

def eval_model(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.eval()
  dynamo_resnet18 = torch.compile(
      xla_resnet18, backend='torchxla_trace_once')
  for data, _ in loader:
    output = dynamo_resnet18(data)
```
> **NOTE:** inference backend name `torchxla_trace_once` is subject to change.

With the `torch.compile` you will see that PyTorch/XLA only traces the resent18 model once during the init time and executes the compiled binary everytime `dynamo_resnet18` is invoked, instead of tracing the model every time. Note that currently Dynamo does not support fallback so if there is an op that can not be traced by XLA, it will error out. Feel free to open a github issue and we We prioritize adding the lowering. We will add the fallback support for dynamo in the upcoming 2.1 release. Here is a inference speed analysis to compare Dynamo and Lazy using torch bench on Cloud TPU v4-8

| model | Speed up |
| --- | ----------- |
resnet18 | 1.768
resnet50 | 1.61
resnext50_32x4d	| 1.328
alexnet | 1.261
mobilenet_v2 | 2.017
mnasnet1_0 | 1.686
vgg16 | 1.155
BERT_pytorch | 3.502
squeezenet1_1 | 1.674
timm_vision_transformer | 3.138
average | 1.9139

### Training
PyTorch/XLA also supports Dynamo for training, but it is very experimental and we are working with the PyTorch Compiler team to iterate on the implementation. On the 2.0 release it only supports forward and backward pass but not the optimizer. Here is an example of training a resnet18 with `torch.compile`

```python
import torch
imprt torchvision
import torch_xla.core.xla_model as xm

def train_model(model, data, target):
  loss_fn = torch.nn.CrossEntropyLoss()
  pred = model(data)
  loss = loss_fn(pred, target)
  loss.backward()
  return pred

def train_model_main(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.train()
  dynamo_train_model = torch.compile(
        train_model, backend='aot_torchxla_trace_once')
  for data, target in loader:
    output = dynamo_train_model(xla_resnet18, data, target)
```

> **NOTE:** Backend we used here is `aot_torchxla_trace_once`(subject to change) instead of `torchxla_trace_once`

We expect to extract and execute 3 graphs per training step instead of one training step if you use the Lazy tensor. Here is a training speed analysis to compare Dynamo and Lazy using a torch bench on Cloud TPU v4-8.

| model | Speed up |
| --- | ----------- |
resnet50 | 0.937
resnet18 | 1.003
BERT_pytorch | 1.869
resnext50_32x4d | 1.139
alexnet | 0.802
mobilenet_v2 | 0.672
mnasnet1_0 | 0.967
vgg16 | 0.742
timm_vision_transformer | 1.69
squeezenet1_1 | 0.958
average | 1.0779

> **NOTE:** We run each model's fwd and bwd for a single step and then collect the e2e time. In the real world we will run multiple steps at each training job which can easily hide the tracing cost from execution(since it is async). Lazy Tensor will have much better performance in that scenario.

We are currently working on the optimizer support and that will be availiable on nightly soon but won't be in the 2.0 release.

### Feature gaps
There are few gaps we want to call out that are preventing us from using the TorchDynamo on larger scale models.

1. TorchDynamo does not trace the communication ops(like `all_reduce` and `all_gather`) yet. It will let the communication ops run on eager mode, which is not ideal for PyTorch/XLA use case since we would want the compiler to see all communication ops in the same graph.
2. TorchDynamo will trace forward and backward into separate graphs. For PyTorch/XLA it is important to let the XLA compiler see the whole step as one graph to best optimize the speed. There is also a fixed overhead to launch every device execution which make executing multiple graphs per training step less ideal.

These feature gaps compared to Lazy Tensor makes it less efficient in real world training use cases, especially the tracing cost can be overlapped with the execution in training.

### Take away
TorchDynamo provides a really promising way for the compiler backend to hide the complexity from the user and easily retrieve the modeling code in a graph format. Compared with PyTorch/XLA's traditional Lazy Tensor way of extracting the graph, TorchDynamo can skip the graph tracing for every iteration, hence providing a much better inference response time.

Most models supported by PyTorch/XLA, have seen significant speedup when running inference with the new dynamo-xla bridge. Our community is working hard to expand the set of supported models. Regarding the training feature gaps mentioned above, the PyTorch/XLA community is super excited to improve the training gap in our upcoming development work. The team continues to heavily invest in TorchDynamo and work with the upstream to mature the training story.