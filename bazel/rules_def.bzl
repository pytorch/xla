"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

def ptxla_cc_library(
        deps = [],
        copts = [],
        **kwargs):
    native.cc_library(
        copts = copts + ["-isystemexternal/torch"],  # Required for system includes.
        deps = deps + [
            "@torch//:headers",
            "@torch//:runtime_headers",
        ],
        **kwargs
    )

def ptxla_cc_test(
        deps,
        copts = [],
        **kwargs):
    tf_cc_test(
        linkstatic = True,
        extra_copts = copts + [
            "-isystemexternal/torch",  # Required for system includes.
            "-fexceptions",  # Required for testing crashes.
        ],
        deps = deps + [
            "@pybind11//:pybind11_embed",  # libpython
            "@torch//:headers",
            "@torch//:libc10",
            "@torch//:libtorch",
            "@torch//:libtorch_cpu",
            "@torch//:libtorch_python",
        ],
        **kwargs
    )

envs = {
    "default": {},
    "dynamic": {"XLA_EXPERIMENTAL": "nonzero:masked_select:masked_scatter"},
    "xla_backend_mp": {"MASTER_ADDR": "localhost", "MASTER_PORT": "6000"},
    "downcast_bf16": {"XLA_DOWNCAST_BF16": "1"},
    "xla_ir_debug": {"XLA_IR_DEBUG": "1"},
    "xla_hlo_debug": {"XLA_HLO_DEBUG": "1"},
    "use_bf16": {"XLA_USE_BF16": "1"},
    "without_functionalization": {"XLA_DISABLE_FUNCTIONALIZATION": "1"},
    "async_scalar": {"XLA_TRANSFER_SCALAR_ASYNC": "1"},
    "opbyop": {"XLA_GET_TENSORS_OPBYOP": "1", "XLA_SYNC_TENSORS_OPBYOP": "1"},
    "eager_debug": {"XLA_USE_EAGER_DEBUG_MODE": "1"},
    "save_tensor_file": {"XLA_SAVE_TENSORS_FILE": "/tmp/xla_test_save_ir.txt"},
}

ACCELERATORS = {
    "cpu": {"PJRT_DEVICE": "CPU"},
    "gpu": {"PJRT_DEVICE": "GPU"},
    "tpu": {"PJRT_DEVICE": "TPU"},
}

def merge(modes):
    z = {}
    for m in modes:
        z.update(envs[m])
    return z

default_env = {
    "VERBOSITY": "2",
    "TRIM_GRAPH_SIZE": "500",
    "TRIM_GRAPH_CHECK_FREQUENCY": "100",
    "PYTORCH_TEST_WITH_SLOW": "1",
    "XLA_DUMP_FATAL_STACK": "1",
}

def union(a, b):
    z = {}
    z.update(default_env)
    z.update(a)
    z.update(b)
    return z

def ptxla_py_test(
        name = "name",
        modes = ["default"],
        accelerators = ["cpu", "gpu", "tpu"],
        deps = [],
        tags = [],
        **kwargs):
    for mode in modes:
        for accelerator in accelerators:
            native.py_test(
                tags = tags + [mode, "py_%s" % accelerator],
                python_version = "PY3",
                deps = deps + ["//test:test_utils"],
                timeout = "eternal",
                main = "%s.py" % name,
                name = "%s_%s_%s" % (name, mode, accelerator),
                env = union(envs[mode], ACCELERATORS[accelerator]),
                **kwargs
            )
