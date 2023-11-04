# Contribute To PyTorch/XLA

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue, please comment on the thread and we're happy to provide any guidance.
You are very welcome to pick issues from [good first issue](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help wanted](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

## Building Manually

We recommend you to use our prebuilt Docker image to start your development work. If you want to use VSCode with docker, please refer to this [config](https://github.com/pytorch/xla/tree/master/.devcontainer/tpu-contributor).

* Setup Development Docker Image

  ```shell
  docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
  docker run --privileged --name ptxla -it -d -e "TERM=xterm-256color" us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
  docker exec --privileged -it ptxla /bin/bash
  ```
  All of the code below will be assumed to be run within the docker.

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```Shell
  git clone --recursive https://github.com/pytorch/xla.git
  ```

* Build PyTorch
  ```Shell
  cd /pytorch/
  python setup.py install
  ```
* Build PyTorch/XLA
  ```Shell
  cd xla/
  python setup.py install
  ```

### Build PyTorch/XLA from source with GPU support

Please refer to this [guide](https://github.com/pytorch/xla/blob/master/docs/gpu.md#develop-pytorchxla-on-a-gpu-instance-build-pytorchxla-from-source-with-gpu-support).

## Before Submitting A Pull Request:

In `pytorch/xla` repo we enforce coding style for both C++ and Python files. Please try to format
your code before submitting a pull request.

### C++ Style Guide

`pytorch/xla` uses `clang-format-7` with a customized style config.
If your PR touches the C++ source files, please run the following command before submitting a PR.

```Shell
# How to install: sudo apt install clang-format-7
# If your PR only changes foo.cpp, run the following in xla/ folder
clang-format-7 -i -style=file /PATH/TO/foo.cpp
# To format all cpp files, run the following in xla/ folder
find -name '*.cpp' -o -name '*.h' -o -name '*.cc' | xargs clang-format-7 -i -style=file
```

### Python Style Guide

`pytorch/xla` uses `yapf`(specially version 0.30.0 in case it's not backward compatible) with a customized style config.
If your PR touches the Python source files, please run the following command before submitting a PR.

```Shell
# How to install: pip install yapf==0.30.0
yapf --recursive -i *.py test/ scripts/ torch_xla/
```

### Running the Tests

To run the tests, follow __one__ of the options below:

* Run on local CPU:

  ```Shell
  export PJRT_DEVICE=CPU
  ```

* Run on Cloud TPU:

  ```Shell
  export PJRT_DEVICE=TPU
  ```

* Run on GPU:

  ```Shell
  export PJRT_DEVICE=CUDA GPU_NUM_DEVICES=${NUM_GPU}
  ```

For more detail on configuring the runtime, please refer to [this doc](https://github.com/pytorch/xla/blob/master/docs/pjrt.md#quickstart)

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.

### Useful materials
1. [OP Lowering Guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md)
2. [CODEGEN MIGRATION GUIDE](https://github.com/pytorch/xla/blob/master/CODEGEN_MIGRATION_GUIDE.md)
