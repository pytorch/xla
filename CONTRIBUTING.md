# Contribute To PyTorch/XLA

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue, please comment on the thread and we're happy to provide any guidance.
You are very welcome to pick issues from [good first issue](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help wanted](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

## Building Manually

To build from source:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```Shell
  git clone --recursive https://github.com/pytorch/xla.git
  ```

### Building Docker Image

* We provide a Dockerfile in `docker/` that you can use to build images as the
  following command:

  ```Shell
  cd xla/
  docker build -t torch-xla -f docker/Dockerfile .
  ```

### Building With Script

* To build and install `torch` and `torch_xla`:

  ```Shell
  xla/scripts/build_torch_wheels.sh
  ```

### Build From Source

* Apply PyTorch patches:

  ```Shell
  xla/scripts/apply_patches.sh
  ```

* Install the Lark parser used for automatic code generation:

  ```Shell
  pip install lark-parser
  ```

* Currently _PyTorch_ does not build with _GCC_ 6.x, 7.x, and 8.x (various kind of ICEs). _CLANG_ 7, 8, 9 and 10 are known to be working, so install that in your VM:

  ```Shell
  sudo apt-get install clang-8 clang++-8
  export CC=clang-8 CXX=clang++-8
  ```

  You may need to add the following line to your _/etc/apt/sources.list_ file:

  ```Shell
  deb http://deb.debian.org/debian/ testing main
  ```

  And run the following command before trying again to install _CLANG_:

  ```Shell
  sudo apt-get update
  ```

* Build _PyTorch_ from source following the regular [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  python setup.py install
  ```

* Install Bazelisk following the [instructions](https://github.com/bazelbuild/bazelisk#requirements). Bazelisk automatically picks a good version of Bazel for PyTorch/XLA build.

* Build the _PyTorch/XLA_ source:

  ```Shell
  cd xla/
  python setup.py install
  ```

## Before Submiting A Pull Request:

In `pytorch/xla` repo we enforce coding style for both C++ and Python files. Please try to format
your code before submitting a pull request.

### C++ Style Guide

`pytorch/xla` uses `clang-format-7` with a customized style config.
If your PR touches the C++ source files, please run the following command before submmiting a PR.

```Shell
# How to install: sudo apt install clang-format-7
# If your PR only changes foo.cpp, run the following in xla/ folder
clang-format-7 -i -style=file /PATH/TO/foo.cpp
# To format all cpp files, run the follwoing in xla/ folder
find -name '*.cpp' -o -name '*.h' | xargs clang-format-7 -i -style=file
```

### Python Style Guide

`pytorch/xla` uses `yapf`(specially version 0.30.0 in case it's not backward compatible) with a customized style config.
If your PR touches the Python source files, please run the following command before submmiting a PR.

```Shell
# How to install: pip install yapf==0.30.0
yapf --recursive -i *.py test/ scripts/ torch_xla/
```

### Running the Tests

To run the tests, follow __one__ of the options below:

* Run on local CPU using the XRT client:

  ```Shell
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  ```

  Select any free TCP port you prefer instead of 40934 (totally arbitrary).

* Run on Cloud TPU using the XRT client, set the XRT_TPU_CONFIG environment variable:

  ```Shell
  export XRT_TPU_CONFIG="localservice;0;localhost:51011"
  ```

Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.

### Useful materials
1. [OP Lowering Guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md)
2. [CODEGEN MIGRATION GUIDE](https://github.com/pytorch/xla/blob/master/CODEGEN_MIGRATION_GUIDE.md)
