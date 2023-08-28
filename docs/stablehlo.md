# StableHLO API in PyTorch/XLA

[StableHLO](https://github.com/openxla/stablehlo) is an operation set for high-level operations (HLO) in machine learning (ML) models. Essentially, it's a portability layer between different ML frameworks and ML compilers.

The APIs are experimental now, please open a GitHub issue to PyTorch/XLA team if there is issues/questions regarding the API.

## Export StableHLO from PyTorch/XLA

```
import torch
import torchvision
import torch_xla.core.xla_model as xm

xla_resnet18 = torchvision.models.resnet18().to(xm.xla_device())
xla_data = torch.randn(4, 3, 224, 224, device=xm.xla_device())
xla_output = xla_resnet18(xla_data)
stablehlo_txt = xm.get_stablehlo([xla_output])
stablehlo_bytecode = xm.get_stablehlo_bytecode([xla_output])
```

PyTorch/XLA provides API to serialize PyTorch program to StableHLO in text or bytecode format. The graph will be traced in PyTorch/XLA by Lazy Tensor Core (LTC). And then lower to StableHLO.

## Execute StableHLO bytecode with XLA backend

```
import torch
import torch_xla.core.xla_model as xm

xla_data = torch.randn(4, 3, 224, 224, device=xm.xla_device())
res = torch_xla._XLAC._run_stablehlo(stablehlo_bytecode, [xla_data])
res = res.cpu()
```

PyTorch/XLA provides API to execute StableHLO bytecode with the `torch.tensor` inputs. The StableHLO bytecode will be compiled by XLA compiler and execute on {CPU, TPU, GPU} (Set environment variable `PJRT_DEVICE=CPU/TPU/GPU` for different backend). The returned result will be XLA Tensors, calling `.cpu()` will convert the XLA tensor to regular CPU Torch Tensors.
