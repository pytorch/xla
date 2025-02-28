# torchax: Running PyTorch on TPU

**torchax!** is a backend for PyTorch, allowing users to run
PyTorch on Google CloudTPUs. **torchax!** is also a library for providing
graph-level interoperability between PyTorch with Jax.

This means, with **torchax** you can:
* Run PyTorch code on TPU with as little as 2 lines of code change.
* Call a jax function from a pytorch function, passing in `jax.Array`s
* Call a pytorch function from a jax function, passing in a `torch.Tensor` subclass.
* Use jax features such as `jax.grad`, `optax` and `GSMPD` to train a Pytorch model.
* Use a Pytorch model as feature extractor and use it with a Jax model.
etc etc.

## Install


### On Google Cloud TPU:
First install torch CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install jax TPU:

```bash
pip install -U jax[tpu]
```

Finally install torchax

```bash
pip install torchax
```

### On GPU machines:
First install torch CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install jax CUDA:

```bash
pip install -U jax[cuda12]
```

Finally install torchax

```bash
pip install torchax
```

### On CPU machines (mac included)
First install torch CPU:

```bash
# Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu

# OR Mac:
pip install torch
```

Then install jax CPU:

```bash
pip install -U jax
```

Finally install torchax

```bash
pip install torchax
```

NOTE: if you like metal support for Apple devices then install the
metal version of jax: https://developer.apple.com/metal/jax/

### Installing `torchax` from source

Still need to install `torch` CPU and `Jax` of your accelerator (GPU, TPU or None).

```bash
pip install git+https://github.com/pytorch/xla.git#subdirectory=torchax
```

## Run a model

Now let's execute a model under torchax. We'll start with a simple 2-layer model
it can be in theory any instance of `torch.nn.Module`.

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

# Execute this model using torch
inputs = torch.randn(3, 3, 28, 28)
print(m(inputs))
```

This model `m` contains 2 parts: the weights that is stored inside of the model
and it's submodules (`nn.Linear`).

To execute this model with `torchax`; we need to enable torchax to capture pytorch ops.
To enable this, use:

```python
import torchax
torchax.enable_globally()
```
Then, a `jax` device will be available to use

```python
inputs = torch.randn(3, 3, 28, 28, device='jax')
m = MyModel()
res = m(inputs)
print(type(res))  # outputs torchax.tensor.Tensor
```

`torchax.tensor.Tensor` is a `torch.Tensor` subclass that holds
a `jax.Array`. You can inspect that jax array with `res.jax()`


## What is happening behind the scene:

We took the approach detailed in [new device](https://github.com/albanD/subclass_zoo/blob/main/new_device.py) recipe by Alban (@albanD); using `jax.Array` for the `raw_data`.

In other words, When a torch op is executed inside of `env` context manager (which is enabled with `torchax.enable_globally()`), we can swap out the
implementation of that op written in Jax.

When a model's constructor runs, it will call some tensor constructor, such as
`torch.rand`, `torch.ones` or `torch.zeros` etc to create its weights. The constructor
will create an `torch.Tensor` subclass that contains a `jax.Array`.

Then, each subsequent op can unpack the `jax.Array`, call the op implementation,
and wraps it back into `torch.Tensor` subclass.

See more at [how_it_works](docs/how_it_works.md) and [ops registry](docs/ops_registry.md).


### Executing with jax.jit

The above script will execute the model using eager mode Jax as backend. This
does allow executing torch models on TPU, but is often slower than what we can
achieve with `jax.jit`.

`jax.jit` is a function that takes a Jax function (i.e. a function that takes jax array
and returns jax array) into the same function, but faster.

We have made the `jax_jit` decorator that would accomplish the same with functions
that takes and returns `torch.Tensor`. To use this, the first step is to create
a functional version of this model: this means the parameters should be passed in
as input instead of being attributes on class:


```python

def model_func(param, inputs):
  return torch.func.functional_call(m, param, inputs)

```
Here we use [torch.func.functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html)
from PyTorch to replace the model
weights with `param`, then call the model. This is roughly equivalent to:

```python
def model_func(param, inputs):
  m.load_state_dict(param)
  return m(*inputs)
```

Now, we can apply `jax_jit`

```python
from torchax.interop import jax_jit
model_func_jitted = jax_jit(model_func)
print(model_func_jitted(new_state_dict, inputs))
```

See more examples at [eager_mode.py](examples/eager_mode.py) and the (examples folder)[examples/]

However, to ease the idiom of creating functional model and calling it with parameters,
we also created the `JittableModule` helper class.

So the above can be written as:

```python

from torchax.interop import JittableModule

m_jitted = JittableModule(m)
res = m_jitted(...)
```

The first time that `m_jitted` is called , it will trigger `jax.jit`
then the subsequent computation with inputs of same shape will be fast.



# Citation:

@software{torchax,
  author = {Han Qi, Chun-nien Chan, Will Cromar, Manfei Bai, Kevin Gleanson},
  title = {torchax: PyTorch on TPU and Jax interoperability},
  url = {https://github.com/pytorch/xla/tree/master/torchax}
  version = {0.0.4},
  date = {2025-02-24},
}

# Maintainers & Contributors:

This library is created and maintained by the PyTorch/XLA team at Google Cloud.

However, it benefitted from many direct and indirect
contributions outside of the team. Many of them done by
fellow Googlers using [Google's 20% project policy](https://ebsedu.org/blog/google-tapping-workplace-actualization-20-time-rule), others by partner teams.

Here is the full list of contributors by 2025-02-25.

Han Qi (qihqi), Pytorch / XLA
Manfei Bai (manfeibai), Pytorch / XLA
Will Cromar (will-cromar), Meta
Milad Mohammadi (miladm), Pytorch / XLA
Siyuan Liu (lsy323), Pytorch / XLA
Bhavya Bahl (bhavya01), Pytorch / XLA
Pei Zhang (zpcore), Pytorch / XLA
Yifei Teng (tengyifei), Pytorch / XLA
Chunnien Chan (chunnienc), Google, ODML
Alban Desmaison (albanD), Meta, Pytorch
Simon Teo (simonteozw), Google(20%)
David Huang (dvhg), Google(20%)
Barni Seetharaman (barney-s), Google(20%)
Anish Karthik (anishfish2) , Google(20%)
Yao Gu (guyao) , Google(20%)
Yenkai Wang (yenkwang) , Google(20%)
Greg Shikhman (commander) , Google(20%)
Matin Akhlaghinia (matinehAkhlaghinia), Google(20%)
Tracy Chen (tracych477), Google(20%)
Matthias Guenther (mrguenther) , Google(20%)
WenXin Dong (wenxindongwork), Google(20%)
Kevin Gleason (GleasonK) , Google, StableHLO
Nupur Baghel (nupurbaghel), Google(20%)
Gwen Mittertreiner (gmittert), Google(20%)
Zeev Melumian (zmelumian), Lightricks
Vyom Sharma (vyom1611), Google(20%)
Shitong Wang (ShitongWang), Adobe
Rémi Doreau (ayshiff), Google(20%)
Lance Wang (wang2yn84), Google, CoreML
Hossein Sarshar (hosseinsarshar) , Google(20%)
Daniel Vega-Myhre (danielvegamyhre) , Google(20%)
Tianqi Fan (tqfan28), Google(20%)
Jim Lin (jimlinntu), Google(20%)
Fanhai Lu (FanhaiLu1), Google Cloud
DeWitt Clinton (dewitt), Google PyTorch
Aman Gupta (aman2930) , Google(20%)