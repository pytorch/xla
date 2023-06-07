"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@tsl//tsl/platform/default:rules_cc.bzl",
    "cc_test",
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