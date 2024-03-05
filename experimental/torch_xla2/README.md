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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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