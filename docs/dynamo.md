## TorchDynamo(torch.compile) integration in PyTorch XLA

[TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html) is a Python-level JIT compiler designed to make unmodified PyTorch programs faster. It provides a clean API for compiler backends to hook in and its biggest feature is to dynamically modify Python bytecode right before it is executed. In the pytorch/xla 2.0 release, PyTorch/XLA provided an experimental backend for the TorchDynamo for both inference and training.

The way that XLA bridge works is that Dynamo will provide a TorchFX graph when it recognizes a model pattern and PyTorch/XLA will use existing Lazy Tensor technology to compile the FX graph and return the compiled function.

### Inference
Here is a small code example of running resnet18 with `torch.compile`

```python
import torch
import torchvision
import torch_xla.core.xla_model as xm

def eval_model(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.eval()
  dynamo_resnet18 = torch.compile(
    xla_resnet18, backend='openxla')
  for data, _ in loader:
    with torch.no_grad():
      output = dynamo_resnet18(data)
```

With the `torch.compile` you will see that PyTorch/XLA only traces the resent18 model once during the init time and executes the compiled binary every time `dynamo_resnet18` is invoked, instead of tracing the model every time. Here is a inference speed analysis to compare Dynamo and Lazy using torch bench on Cloud TPU v4-8

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

Note 
1. User will likely see better inference perfomrance by putting the inference execution in a `torch.no_grad` context. `openxla` is a `aot-autograd` backend of `torch.compile`. `Aot-autograd` will attempt to save some states for potential backward. `torch.no_grad` will help `aot-autograd` understand that it is being executed in a inference context.
2. User can also use the `openxla_eval` backend directly without `torch.no_grad`, since `openxla_eval` is not an `aot-autograd` backend and only works for inference. 

### Training
PyTorch/XLA also supports Dynamo for training, but it is  experimental and we are working with the PyTorch Compiler team to iterate on the implementation. Here is an example of training a resnet18 with `torch.compile`

```python
import torch
import torchvision
import torch_xla.core.xla_model as xm

def train_model(model, data, target, optimizer):
  loss_fn = torch.nn.CrossEntropyLoss()
  pred = model(data)
  loss = loss_fn(pred, target)
  loss.backward()
  optimizer.step()
  return pred

def train_model_main(loader):
  device = xm.xla_device()
  xla_resnet18 = torchvision.models.resnet18().to(device)
  xla_resnet18.train()
  dynamo_train_model = torch.compile(
        train_model, backend='openxla')
  for data, target in loader:
    xla_optimizer = optim.SGD(data, lr=0.1, weight_decay=1e-2)
    output = dynamo_train_model(xla_resnet18, data, target, xla_optimizer)
```

We expect to extract and execute 3 graphs per training step instead of 1 graph per training step if you use the Lazy tensor. Here is a training speed analysis to compare Dynamo and Lazy using a torch bench on Cloud TPU v4-8.

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

### Feature gaps
There is one gap we want to call out that are preventing us from using the TorchDynamo on larger scale models.

1. TorchDynamo will trace forward and backward into separate graphs. For PyTorch/XLA it is important to let the XLA compiler see the whole step as one graph to best optimize the speed. There is also a fixed overhead to launch every device execution which make executing multiple graphs per training step less ideal.

This gap compared to Lazy Tensor makes it less efficient in real world training use cases, especially the tracing cost can be overlapped with the execution in training.

### Take away
TorchDynamo provides a really promising way for the compiler backend to hide the complexity from the user and easily retrieve the modeling code in a graph format. Compared with PyTorch/XLA's traditional Lazy Tensor way of extracting the graph, TorchDynamo can skip the graph tracing for every iteration, hence providing a much better inference response time.

Most models supported by PyTorch/XLA, have seen significant speedup when running inference with the new dynamo-xla bridge. Our community is working hard to expand the set of supported models. Regarding the training feature gaps mentioned above, the PyTorch/XLA community is super excited to improve the training gap in our upcoming development work. The team continues to heavily invest in TorchDynamo and work with the upstream to mature the training story.
