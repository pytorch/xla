# Custom Hardware Plugins

PyTorch/XLA supports custom hardware through OpenXLA's PJRT C API. The
PyTorch/XLA team directly supports plugins for Cloud TPU (`libtpu`) and
GPU ([OpenXLA](https://github.com/openxla/xla/tree/main/xla/pjrt/gpu)).
The same plugins may also be used by JAX and TF.

## Implementing a PJRT Plugin

PJRT C API plugins may be closed-source or open-source. They contain two
parts:

1.  Binary exposing a PJRT C API implementation. This part can be shared
    with JAX and TensorFlow.
2.  Python package containing the above binary, as well as an
    implementation of our `DevicePlugin` Python interface, which handles
    additional setup.

### PJRT C API Implementation

In short, you must implement a
[PjRtClient](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h)
containing an XLA compiler and runtime for your device. The PJRT C++
interface is mirrored in C in the
[PJRT_Api](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h).
The most straightforward option is to implement your plugin in C++ and
[wrap
it](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_wrapper_impl.h)
as a C API implementation. This process is explained in detail in
[OpenXLA's
documentation](https://openxla.org/xla/pjrt_integration#how_to_integrate_with_pjrt).

For a concrete example, see the [example
implementation](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_cpu_internal.cpp).

### PyTorch/XLA Plugin Package

At this point, you should have a functional PJRT plugin binary, which
you can test with the placeholder `LIBRARY` device type. For example:

    $ PJRT_DEVICE=LIBRARY PJRT_LIBRARY_PATH=/path/to/your/plugin.so python
    >>> import torch_xla
    >>> torch_xla.devices()
    # Assuming there are 4 devices. Your hardware may differ.
    [device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]

To register your device type automatically for users as well as to
handle extra setup for e.g. multiprocessing, you may implement the
`DevicePlugin` Python API. PyTorch/XLA plugin packages contain two key
components:

1.  An implementation of `DevicePlugin` that (at the very least)
    provides the path to your plugin binary. For example:

``` python
class CpuPlugin(plugins.DevicePlugin):

  def library_path(self) -> str:
    return os.path.join(
        os.path.dirname(__file__), 'lib', 'pjrt_c_api_cpu_plugin.so')
```

2.  A `torch_xla.plugins` [entry
    point](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)
    that identifies your `DevicePlugin`. For exmaple, to register the
    `EXAMPLE` device type in a `pyproject.toml`:

```{=html}
<!-- -->
```
    [project.entry-points."torch_xla.plugins"]
    example = "torch_xla_cpu_plugin:CpuPlugin"

With your package installed, you may then use your `EXAMPLE` device
directly:

    $ PJRT_DEVICE=EXAMPLE python
    >>> import torch_xla
    >>> torch_xla.devices()
    [device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]

[DevicePlugin](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/plugins.py)
provides additional extension points for multiprocess initialization and
client options. The API is currently in an experimental state, but it is
expected to become stable in a future release.
