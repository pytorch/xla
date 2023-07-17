"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@xla//xla:xla.bzl",
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
