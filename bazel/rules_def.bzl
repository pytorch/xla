"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "@org_tensorflow//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//bazel:tensorflow.bzl",
    "if_with_tpu_support",
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
        ] + if_cuda_is_configured([
            "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_device",
            "@org_tensorflow//tensorflow/compiler/xla/stream_executor:cuda_platform",
        ]) + if_with_tpu_support([
            "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_device",
            "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_jit",
        ]),
        **kwargs
    )
