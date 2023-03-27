## Bazel in Pytorch/XLA

[Bazel](https://bazel.build/) is a free software tool used for the automation of building and testing software. [TensorFlow](https://www.tensorflow.org/http) and [OpenXLA](https://github.com/openxla/xla) both use it, which makes it a good fit for PyTorch/XLA as well.

## Bazel dependencies

Tensorflow is a [bazel external dependency](https://bazel.build/external/overview) for PyTorch/XLA, which can be seen in the `WORKSPACE` file:

`WORKSPACE`
```bzl
http_archive(
    name = "org_tensorflow",
    patch_args = [ "-l", "-p1"],
    patch_tool = "patch",
    patches = [
        "//tf_patches:thread_local_random.diff",
        "//tf_patches:xplane.diff",
        ...
    ],
    strip_prefix = "tensorflow-f7759359f8420d3ca7b9fd19493f2a01bd47b4ef",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/f7759359f8420d3ca7b9fd19493f2a01bd47b4ef.tar.gz",
    ],
)
```

TensorFlow pin can be updated by pointing this repository to a different revision. Patches may be added as needed.
Bazel will resolve the dependency, prepare the code and patch it hermetically.

For PyTorch, a different dependency mechanism is deployed because a local [PyTorch](https://github.com/pytorch/pytorch)
checkout is used, and this local checkout has to be `built` from source and ideally installed on the system for version
compatibility (e.g. codegen in PyTorch/XLA uses `torchgen` python module that should be installed in the system).

The local directory can either set in `bazel/dependencies.bzl`, or overriden on the command line:

```bash
bazel build --override_repository=org_tensorflow=/path/to/exported/tf_repo //...
```

```bash
bazel build --override_repository=torch=/path/to/exported/and/built/torch_repo //...
```

Please make sure that the overridden repositories are at the appropriate revisions and in case of `torch`, that it
has been built with `USE_CUDA=0 python setup.py bdist_wheel` to make sure that all expected build objects are present;
ideally installed into the system.

`WORKSPACE`
```bzl
new_local_repository(
    name = "torch",
    build_file = "//bazel:torch.BUILD",
    path = PYTORCH_LOCAL_DIR,
)
```

PyTorch headers are directly sourced from the `torch` dependency, the local checkout of PyTorch. The shared libraries
(e.g. `libtorch.so`) are sourced from the same local checkout where the code has been built and `build/lib/` contains the
built objects. For this to work, it's required to pass `-isystemexternal/torch` to the compiler so it can find `system` libraries and satisfy them from the local checkout. Some are included as `<system>` and some as `"user"` headers.

Bazel brings in [pybind11](https://github.com/pybind/pybind11) embeded python and links against it to provide `libpython`
to the plugin using this mechanism. Python headers are also sourced from there instead of depending on the system version.
These are satisfied from the `"@pybind11//:pybind11_embed"`, which sets up compiler options for linking with `libpython`
transitively.

## How to build XLA libraries

Building the libraries is simple:

```bash
bazel build //third_party/xla_client/...
```

Bazel is configred via `.bazelrc`, but it can also take flags on the command line.

```bash
bazel build --config=remote_cache //third_party/xla_client/...
```

The `remote_cache` configurations use gcloud for caching and  usually faster, but require
authentication with gcloud. See `.bazelrc` for the configuration.

Using bazel makes it easy to express complex dependencies and there is a lot of gain from having a single build graph
with everything expressed in the same way. Therefore, there is no need to build the XLA libraries separately from the
rest of the pluing as used to be the case, building the whole repository, or the plugin shared object that links everythin
else in, is enough.

## How to build the Torch/XLA plugin

The normal build can be achieved by the invoking the standard `python setup.py bdist_wheel`, but C++ bindings can be built simply with:

```bash
bazel build //:_XLAC.so
```

This will build the XLA client and the PyTorch plugin and link it all together. This can be useful when testing changes, to be able to compile the C++ code without building the python plugin faster iteration cycles.

## Remote caching

Bazel comes with [remote caching](https://bazel.build/remote/caching) built in. There are plenty of cache backends that can be used; we deploy our caching on (GCS)[https://bazel.build/remote/caching#cloud-storage]. You can see the configuration in `.bazelrc`, under config name `remote_cache`.

Remote caching is disabled by default but because it speeds up incremental builds by a huge margin, it is almost always recommended, and it is enabled by default in the CI automation and on Cloud Build.

Using the remote cache configured by `remote_cache` configuration setup requires authentication with GCP. There are various ways to authenticate with GCP. For individual developers who have access to the development GCP project, one only needs to
specify the `--config=remote_cache` flag to bazel, and the default `--google_default_credentials` will be used and if the
gcloud token is present on the machine, it will work out of the box, using the logged in user for authentication. The user
needs to have remote build permissions in GCP. In the CI, the service account key is used for authentication and is passed to
bazel using `--config=remote_cache --google_credentials=path/to/service.key`. On [Cloud Build](https://cloud.google.com/build), `docker build --network=cloudbuild` is used to pass the authentication from the service account running the cloud build down into the docker image doing the compilation: [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) does the work there and authenticates as the service account. All accounts, both
user and service accounts, need to have remote cache read/write permissions.

Remote cache uses cache silos. Each unique machine and build should specify a unique silo key to benefit from consistent caching. The silo key can be passed using a flag: `-remote_default_exec_properties=cache-silo-key=SOME_SILO_KEY'`.

Running the build with remote cache:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" TPUVM_MODE=1 python setup.py bdist_wheel
```

`YOUR-USER` here can the author's username or machine name, a unique name that ensures good cache behavior. Other `setup.py` functionality works as intended too (e.g. `develop`).

The first time the code is compiled using a new cached key will be slow because it will compile everything from scratch, but incremental compilations will be very fast. On updating the TensorFlow pin, it will once again be a bit slower the first time per key, and then until the next update quite fast again.

## Running tests

Currently C++ code is built and tested by bazel. Python code will be migrated in the future.

Bazel is a test plafrom too, making it easy to run tests:

```bash
bazel test //test/cpp:main
```

Ofcourse the XLA and PJRT configuration have to be present in the environment to run the tests. Not all environmental variables are passed into the bazel test environment to make sure that the remote cache misses are not too common (environment
is part of the cache key), see `.bazelrc` test configuration to see which ones are passed in, and add new ones as required.

You can run the tests using the helper script too:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" ./test/cpp/run_tests.sh -R
```

The `xla_client` tests are pure hermetic tests that can be easily executed. The `torch_xla` plugin tests are more complex:
they require `torch` and `torch_xla` to be installed, and they cannot run in parallel, since they are using either
XRT server/client on the same port, or because they use a GPU or TPU device and there's only one available at the time.
For that reason, all tests under `torch_xla/csrc/` are bundled into a single target `:main` that runs them all sequentially.

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

As always, PyTorch/XLA can be built using Python `distutils`:

```bash
BAZEL_REMOTE_CACHE=1 SILO_NAME="cache-silo-YOUR-USER" TPUVM_MODE=1 python setup.py bdist_wheel
```