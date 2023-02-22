## Bazel in Pytorch/XLA

[Bazel](https://bazel.build/) is a free software tool used for the automation of building and testing software. [TensorFlow](https://www.tensorflow.org/http) and [OpenXLA](https://github.com/openxla/xla) both use it, which makes it a good fit for PyTorch/XLA as well.

## How to build XLA libraries

Building the libraries is simple:

```bash
bazel build //third_party/xla_client/...
```

Bazel is configred via `.bazelrc`, but it can also take flags on the command line.

```bash
bazel build --config=rbe_cpu_linux_py39 //third_party/xla_client/...
bazel build --config=rbe_linux_cuda11.8_nvcc_py3.9 //third_party/xla_client/...
```

The `rbe` configurations build remotely and are cahced and usually faster, but require
authentication with gcloud. See `.bazelrc` for other `rbe` options.

## Running tests

Bazel is a test plafrom too, making it easy to run tests:

```bash
bazel test //third_party/xla_client/...
```

Tests can also be executed remotely in the same fashion as builds.

## Code coverage

When running tests, it can be useful to calculate code coverage.

```bash
bazel coverage //third_party/xla_client/...
```

Coverage can be visualized using `lcov` as described in [Bazel's documentation](https://bazel.build/configure/coverage), or in your editor of choice with lcov plugins, e.g. [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) for VSCode.


## Language Server

Bazel can power a language server like [clangd](https://clangd.llvm.org/) that brings code references, autocompletion and semantic understanding of the underlying code to your editor of choice. For VSCode,
one can use [Bazel Stack](https://github.com/stackb/bazel-stack-vscode-cc) that can be combined with
[clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) functionality to bring powerful features to assist code editing.

## Building PyTorch/XLA

As always, PyTorch/XLA is built using Python `distutils`:

```bash
BAZEL_REMOTE=1 TPUVM_MODE=1 python setup.py bdist_wheel
```

You can build locally without `BAZEL_REMOTE` set too.