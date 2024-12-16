# torchxla2

## Install

Currently this is only source-installable. Requires Python version >= 3.10.

### NOTE:

Please don't install torch-xla from instructions in
https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md .
In particular, the following are not needed:

* There is no need to build pytorch/pytorch from source.
* There is no need to clone pytorch/xla project inside of pytorch/pytorch
  git checkout.


TorchXLA2 and torch-xla have different installation instructions, please follow
the instructions below from scratch (fresh venv / conda environment.)


### 1. Installing `torch_xla2`

The following instructions assume you are in the `torch_xla2` directory:

```
Fork the repository
$ git clone https://github.com/<github_username>/xla.git
$ cd xla/experimental/torch_xla2
```


#### 1.0 (recommended) Make a virtualenv / conda env

If you are using VSCode, then [you can create a new environment from
UI](https://code.visualstudio.com/docs/python/environments). Select the
`dev-requirements.txt` when asked to install project dependencies.

Otherwise create a new environment from the command line.

```bash
# Option 1: venv
python -m venv my_venv
source my_venv/bin/activate

# Option 2: conda
conda create --name <your_name> python=3.10
conda activate <your_name>

# Either way, install the dev requirements.
pip install -r dev-requirements.txt
```

Note: `dev-requirements.txt` will install the CPU-only version of PyTorch.

#### 1.1 Install this package

If you want to install torch_xla2 without the jax dependency and use the jax dependency from torch_xla:
```bash
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install -e .
```

Otherwise, install `torch_xla2` from source for your platform:
```bash
pip install -e .[cpu]
pip install -e .[cuda]
pip install -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

#### 1.2 (optional) verify installation by running tests

```bash
pip install -r test-requirements.txt
pytest test
```

## Run a model

Now let's execute a model under torch_xla2. We'll start with a simple 2-layer model
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

To execute this model with `torch_xla2`; we need construct and run the model
under an `environment` that captures pytorch ops and swaps them with TPU equivalent.

To create this environment: use

```python
import torch_xla2

env = torch_xla2.default_env() 
```
Then, execute the instantiation of the model, as well as evaluation of model, 
using `env` as a context manager:

```python
with env:
  inputs = torch.randn(3, 3, 28, 28)
  m = MyModel()
  res = m(inputs)
  print(type(res))  # outputs XLATensor2
```

You can also enable the environment globally with
```python
import torch_xla2

torch_xla2.enable_globally() 
```

Then everything afterwards is run with XLA.

## What is happening behind the scene:

When a torch op is executed inside of `env` context manager, we can swap out the 
implementation of that op with a version that runs on TPU. 
When a model's constructor runs, it will call some tensor constructor, such as
`torch.rand`, `torch.ones` or `torch.zeros` etc to create its weights. Those
ops are captured by `env` too and placed directly on TPU.

See more at [how_it_works](docs/how_it_works.md) and [ops registry](docs/ops_registry.md).

### What if I created model outside of `env`.

So if you have

```
m = MyModel()
```
outside of env, then regular torch ops will run when creating this model.
Then presumably the model's weights will be on CPU (as instances of `torch.Tensor`).

To move this model into XLA device, one can use `env.to_xla()` function.

i.e.
```
m2 = env.to_xla(m)
inputs = env.to_xla(inputs)

with env:
  res = m2(inputs)
```

NOTE that we also need to move inputs to xla using `.to_xla`. 
`to_xla` works with all pytrees of `torch.Tensor`.


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
weights with `param`, then call the model. This is equivalent to:

```python
def model_func(param, inputs):
  m.load_state_dict(param)
  return m(*inputs)
```

Now, we can apply `jax_jit`

```python
from torch_xla2.interop import jax_jit
model_func_jitted = jax_jit(model_func)
print(model_func_jitted(new_state_dict, inputs))
```

See more examples at [eager_mode.py](examples/eager_mode.py) and the (examples folder)[examples/]