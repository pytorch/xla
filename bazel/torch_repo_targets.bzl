"""Handles the loading phase to define targets for torch_repo."""

cc_library = native.cc_library

def define_torch_targets():
    cc_library(
        name = "headers",
        hdrs = native.glob(
            ["torch/include/**/*.h"],
            ["torch/include/google/protobuf/**/*.h"],
        ),
        strip_include_prefix = "torch/include",
    )

    # Runtime headers, for importing <torch/torch.h>.
    cc_library(
        name = "runtime_headers",
        hdrs = native.glob(["torch/include/torch/csrc/api/include/**/*.h"]),
        strip_include_prefix = "torch/include/torch/csrc/api/include",
    )

    native.filegroup(
        name = "torchgen_deps",
        srcs = [
            # torchgen/packaged/ instead of aten/src
            "torchgen/packaged/ATen/native/native_functions.yaml",
            "torchgen/packaged/ATen/native/tags.yaml",
            ##"torchgen/packaged/ATen/native/ts_native_functions.yaml",
            "torchgen/packaged/ATen/templates/DispatchKeyNativeFunctions.cpp",
            "torchgen/packaged/ATen/templates/DispatchKeyNativeFunctions.h",
            "torchgen/packaged/ATen/templates/LazyIr.h",
            "torchgen/packaged/ATen/templates/LazyNonNativeIr.h",
            "torchgen/packaged/ATen/templates/RegisterDispatchDefinitions.ini",
            "torchgen/packaged/ATen/templates/RegisterDispatchKey.cpp",
            # Add torch/include prefix
            "torch/include/torch/csrc/lazy/core/shape_inference.h",
            ##"torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
        ],
    )

    # Changed to cc_library from cc_import

    cc_library(
        name = "libtorch",
        srcs = ["torch/lib/libtorch.so"],
    )

    cc_library(
        name = "libtorch_cpu",
        srcs = ["torch/lib/libtorch_cpu.so"],
    )

    cc_library(
        name = "libtorch_python",
        srcs = [
            "torch/lib/libshm.so",  # libtorch_python.so depends on this
            "torch/lib/libtorch_python.so",
        ],
    )

    cc_library(
        name = "libc10",
        srcs = ["torch/lib/libc10.so"],
    )
