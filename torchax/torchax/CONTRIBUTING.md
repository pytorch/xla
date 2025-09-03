# Contributing to torchax

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue, please comment on the thread and we're happy to provide any guidance. You are very welcome to pick issues from good first issue and help wanted labels.


# Developer setup

## Mac setup:
@qihqi

I am able to develop directly on mac (m1) laptop for most of parts. Using steps
in README.md works. The condensed version for easy copy & paste:

```bash
conda create --name <your_name> python=3.10
conda activate <your_name>
pip install --upgrade "jax[cpu]" torch
pip install -r test_requirements.txt
pip install -e .
pip install pytest-xdist # recommended for running test faster
pytest -n auto test
```

## Setup on GPU or TPU

Same as Mac setup, except, if you run test using pytest, please also
add `JAX_PLATFORMS=cpu`. The reason is because pytest usually runs
test in multiple threads. CPU device can be accessed concurrently where
TPU devices usually only allow one accesor per process; so it could deadlock.

### VSCode

I use vscode on my Mac. I loosely followed instruction in
https://code.visualstudio.com/docs/python/python-tutorial
to setup a proper python environment.

The plugins I installed (a subset of the ones listed above) are:
* VSCode's official Python plugin
* Ruff formatter
* Python Debugger

I also changed Python interpreter to point at the one in my conda env.
That is all the changes I have.
