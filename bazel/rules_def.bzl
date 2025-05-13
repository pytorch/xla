"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@xla//xla:xla.default.bzl",
    "xla_cc_test",
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
    xla_cc_test(
        linkstatic = True,
        copts = copts + [
            "-isystemexternal/torch",  # Required for system includes.
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
