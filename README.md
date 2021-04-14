# Lazy Tensors XLA plugin

1. Follow the [instructions](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md) for the lazy tensor core.
1. Clone under your copy of the PyTorch repository: `git clone --recursive https://github.com/pytorch/xla.git` and switch to the `asuhan/xla_ltc_plugin` branch: `git checkout asuhan/xla_ltc_plugin`.
1. Install glob2 and the Lark parser, used for automatic code generation:

```bash
pip install glob2 lark-parser
```

1. Run `python setup.py develop`.
2. Set the environment variables which control the XLA backend, for example:

```bash
export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:40935"
```

3. Run `example.py`.

Suggested build environment:

```bash
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
```

If you want to debug it as well:

```bash
export DEBUG=1
```
