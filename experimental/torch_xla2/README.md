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

#### 1.0 (recommended) Make a virtualenv / conda env

If you are using VSCode, then [you can create a new environment from
UI](https://code.visualstudio.com/docs/python/environments). Select the
`dev-requirements.txt` when asked to install project dependencies.

Otherwise create a new environment from the command line.

```bash
# Option 1: venv
python -m venv create my_venv
source my_venv/bin/activate

# Option 2: conda
conda create --name <your_name> python=3.10
conda activate <your_name>

# Either way, install the dev requirements.
pip install -r dev-requirements.txt
```

Note: `dev-requirements.txt` will install the CPU-only version of PyTorch.

#### 1.1 Install this package

Install `torch_xla2` from source for your platform:

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
