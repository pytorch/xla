# CUDA PJRT plugin (experimental)

This directory contains an experimental implementation of the PJRT GPU client as
a plugin. The actual implementation of the PJRT C API lives in the main OpenXLA
repository (see `bazel build` command below).

## Building

See our [contributing guide](../../CONTRIBUTING.md) for build environment setup
steps.

```bash
# Build wheel
pip wheel plugins/cuda -v
# Or install directly
pip install plugins/cuda -v
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
plugins.register_plugin('CUDA', torch_xla_cuda_plugin.CudaPlugin())
xr.set_device_type('CUDA')

print(xm.xla_device())
```
