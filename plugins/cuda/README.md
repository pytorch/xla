# CUDA PJRT plugin (experimental)

This directory contains an experimental implementation of the PJRT GPU client as
a plugin. The actual implementation of the PJRT C API lives in the main OpenXLA
repository (see `bazel build` command below).

## Building

```bash
# Build PJRT plugin
bazel build @xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1  --config=cuda
# Copy to package dir
cp bazel-bin/external/xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so plugins/cuda/torch_xla_cuda_plugin

# Build wheel
pip wheel plugins/cuda
# Or install directly
pip install plugins/cuda
```

## Usage

```python
import os

# Log device type
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_VMODULE'] = 'pjrt_registry=5'

from torch_xla.experimental import plugins
import torch_xla_cuda_plugin
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Use dynamic plugin instead of built-in CUDA support
plugins.use_dynamic_plugins()
plugins.register_plugin('CUDA', torch_xla_cuda_plugin.GpuPlugin())
xr.set_device_type('CUDA')

print(xm.xla_device())
```
