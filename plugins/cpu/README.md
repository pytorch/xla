# CPU PJRT Plugin (testing)

This directory contains an experimental implementation of the PJRT CPU client as
a plugin. This plugin is for testing only and is not officially supported. Use
`PJRT_DEVICE=CPU` with any PyTorch/XLA installation to use built-in CPU support.

The actual implementation of the PJRT C API lives in the main OpenXLA
repository (see `bazel build` command below).

## Building

See our [contributing guide](../../CONTRIBUTING.md) for build environment setup
steps.

```bash
# Build wheel
pip wheel plugins/cpu -v
# Or install directly
pip install plugins/cpu -v
```

## Usage

```python
import os

# Log device type
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_VMODULE'] = 'pjrt_registry=5'

from torch_xla.experimental import plugins
import torch_xla_cpu_plugin
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Use dynamic plugin instead of built-in CPU support
plugins.use_dynamic_plugins()
plugins.register_plugin('CPU', torch_xla_cpu_plugin.CpuPlugin())
xr.set_device_type('CPU')

print(xm.xla_device())
```
