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


### 1. Install dependencies

#### 1.0 (optional) Make a virtualenv / conda env, and activate it.

```bash
conda create --name <your_name> python=3.10
conda activate <your_name>
```
Or,
```bash
python -m venv create my_venv
source my_venv/bin/activate
```

#### 1.1 Install torch CPU, even if your device has GPU or TPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Or, follow official instructions in [pytorch.org](https://pytorch.org/get-started/locally/) to install for your OS.

#### 1.2 Install Jax for either GPU or TPU

If you are using Google Cloud TPU, then
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

If you are using a machine with NVidia GPU:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you are using a CPU-only machine:
```bash
pip install --upgrade "jax[cpu]"
```

Or, follow the official instructions in https://jax.readthedocs.io/en/latest/installation.html to install for your OS or Device.

#### 1.3 Install this package

```bash
pip install -e .
```

#### 1.4 (optional) verify installation by running tests

```bash
pip install -r test_requirements.txt
pytest test
```


## Run a model

Now let's execute a model under torch_xla2. We'll start with a simple 2-layer model
it can be in theory any instance of `torch.nn.Module`.

```python

import torch_xla2
from torch import nn

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
inputs = (torch.randn(3, 3, 28, 28), )
print(m(*inputs))
```

This model `m` contains 2 parts: the weights that is stored inside of the model
and it's submodules (`nn.Linear`).

To execute this model with `torch_xla2`; we need to move the tensors involved in compute
to `XLA` devices. This can be accomplished with `torch_xla2.tensor.move_to_device`.

We need move both the weights and the input to xla devices:

```python
from torch.utils import _pytree as pytree
from torch_xla2.tensor import move_to_device

inputs = move_to_device(inputs)
new_state_dict = pytree.tree_map_only(torch.Tensor, move_to_device, m.state_dict())
m.load_state_dict(new_state_dict, assign=True)

res = m(*inputs)

print(type(res))  # outputs XLATensor2
```

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
from torch_xla2.extra import jax_jit
model_func_jitted = jax_jit(model_func)
print(model_func_jitted(new_state_dict, inputs))
```


