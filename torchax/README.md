# torchax: Running PyTorch on TPU via JAX

**torchax** is a backend for PyTorch, allowing users to run
PyTorch on Google Cloud TPUs. **torchax** is also a library for providing
graph-level interoperability between PyTorch and JAX.

This means, with **torchax** you can:
* Run PyTorch code on TPUs with as little as 2 lines of code change.
* Call a JAX function from a PyTorch function, passing in `jax.Array`s.
* Call a PyTorch function from a JAX function, passing in a `torch.Tensor`s.
* Use JAX features such as `jax.grad`, `optax`, and `GSPMD` to train a PyTorch
  model.
* Use a PyTorch model as feature extractor and use it with a JAX model.
etc etc.

## Install

First install torch CPU:

```bash
# On Linux.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or on Mac.
pip install torch
```

Then install JAX for the accelerator you want to use:

```bash
# On Google Cloud TPU.
pip install -U jax[tpu]

# Or, on GPU machines.
pip install -U jax[cuda12]

# Or, on Linux CPU machines or Macs (see the note below).
pip install -U jax
```

NOTE: if you like metal support for Apple devices then install the
metal version of JAX: https://developer.apple.com/metal/jax/

Finally install torchax:

```bash
# Install pre-built torchax.
pip install torchax

# Or, install torchax from source.
pip install git+https://github.com/pytorch/xla.git#subdirectory=torchax
```

## Run a model

Now let's execute a model under torchax. We'll start with a simple 2-layer model.
In theory, we can use any instance of `torch.nn.Module`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

m = MyModel()

# Execute this model using torch.
inputs = torch.randn(3, 3, 28, 28)
print(m(inputs))
```

To execute this model with `torchax`, we need to enable torchax to capture PyTorch ops:

```python
import torchax
torchax.enable_globally()
```

Then, we can use a `jax` device:

```python
inputs = torch.randn(3, 3, 28, 28, device='jax')
m = MyModel().to('jax')
res = m(inputs)
print(type(res))  # outputs torchax.tensor.Tensor
```

`torchax.tensor.Tensor` is a `torch.Tensor` subclass that holds
a `jax.Array`. You can inspect that JAX array with `res.jax()`.

## What is happening behind the scene

We took the approach detailed in the
[new device](https://github.com/albanD/subclass_zoo/blob/main/new_device.py)
recipe by Alban (@albanD), using `jax.Array` for `raw_data`.

In other words, when a torch op is executed inside an `env` context manager,
which is enabled by `torchax.enable_globally()`, we will swap out the
implementation of that op with JAX.

When a model's constructor runs, it will call some tensor constructor, such as
`torch.rand`, `torch.ones`, or `torch.zeros` to create its weights. When torchax
is enabled, these constructors will create a `torchax.tensor.Tensor`, which
contains a `jax.Array`.

Then, each subsequent op will extract the `jax.Array`, call the op's JAX
implementation, and wrap the result back into a `torchax.tensor.Tensor`,

See more at [how it works](docs/how_it_works.md) and\
[ops registry](docs/ops_registry.md).

### Executing with jax.jit

The above script will execute the model using eager mode JAX as the backend. This
does allow executing torch models on TPUs, but is often slower than what we can
achieve with `jax.jit`.

`jax.jit` is a function that takes a JAX function (i.e. a function that takes JAX arrays
and returns JAX arrays) into a compiled (thus faster) version of the same function.

We have made a `jax_jit` decorator that would accomplish the same with functions
that takes and returns `torch.Tensor`s. To use this, the first step is to create
a functional version of this model: this means the parameters should be passed in
as input instead of being attributes of the class:

```python
def model_func(param, inputs):
  return torch.func.functional_call(m, param, inputs)
```

Here we use [torch.func.functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html)
from PyTorch to replace the model weights with `param` and then call the
model. This is roughly equivalent to:

```python
def model_func(param, inputs):
  m.load_state_dict(param)
  return m(*inputs)
```

Now, we can apply `jax_jit` on `module_func`:

```python
from torchax.interop import jax_jit

model_func_jitted = jax_jit(model_func)
print(model_func_jitted(new_state_dict, inputs))
```

See more examples at [eager_mode.py](examples/eager_mode.py) and the
[examples folder](examples/).

To ease the idiom of creating functional model and calling it with parameters,
we also created the `JittableModule` helper class. It lets us rewrite the
above as:

```python
from torchax.interop import JittableModule

m_jitted = JittableModule(m)
res = m_jitted(...)
```

The first time `m_jitted` is called, it will trigger `jax.jit` to compile the
compile for the given input shapes. Subsequent calls with the same input shapes
will be fast as the compilation is cached.

## Citation

```
@software{torchax,
  author = {Han Qi, Chun-nien Chan, Will Cromar, Manfei Bai, Kevin Gleanson},
  title = {torchax: PyTorch on TPU and JAX interoperability},
  url = {https://github.com/pytorch/xla/tree/master/torchax}
  version = {0.0.4},
  date = {2025-02-24},
}
```

# Maintainers & Contributors:

This library is created and maintained by the PyTorch/XLA team at Google Cloud.

It benefitted from many direct and indirect
contributions outside of the team. Many of them done by
fellow Googlers using [Google's 20% project policy](https://ebsedu.org/blog/google-tapping-workplace-actualization-20-time-rule).
Others by partner teams at Google and other companies.

Here is the list of contributors by 2025-02-25.

```
Han Qi (qihqi), PyTorch/XLA
Manfei Bai (manfeibai), PyTorch/XLA
Will Cromar (will-cromar), Meta
Milad Mohammadi (miladm), PyTorch/XLA
Siyuan Liu (lsy323), PyTorch/XLA
Bhavya Bahl (bhavya01), PyTorch/XLA
Pei Zhang (zpcore), PyTorch/XLA
Yifei Teng (tengyifei), PyTorch/XLA
Chunnien Chan (chunnienc), Google, ODML
Alban Desmaison (albanD), Meta, PyTorch
Simon Teo (simonteozw), Google (20%)
David Huang (dvhg), Google (20%)
Barni Seetharaman (barney-s), Google (20%)
Anish Karthik (anishfish2), Google (20%)
Yao Gu (guyao), Google (20%)
Yenkai Wang (yenkwang), Google (20%)
Greg Shikhman (commander), Google (20%)
Matin Akhlaghinia (matinehAkhlaghinia), Google (20%)
Tracy Chen (tracych477), Google (20%)
Matthias Guenther (mrguenther), Google (20%)
WenXin Dong (wenxindongwork), Google (20%)
Kevin Gleason (GleasonK), Google, StableHLO
Nupur Baghel (nupurbaghel), Google (20%)
Gwen Mittertreiner (gmittert), Google (20%)
Zeev Melumian (zmelumian), Lightricks
Vyom Sharma (vyom1611), Google (20%)
Shitong Wang (ShitongWang), Adobe
Rémi Doreau (ayshiff), Google (20%)
Lance Wang (wang2yn84), Google, CoreML
Hossein Sarshar (hosseinsarshar), Google (20%)
Daniel Vega-Myhre (danielvegamyhre), Google (20%)
Tianqi Fan (tqfan28), Google (20%)
Jim Lin (jimlinntu), Google (20%)
Fanhai Lu (FanhaiLu1), Google Cloud
DeWitt Clinton (dewitt), Google PyTorch
Aman Gupta (aman2930), Google (20%)
```
