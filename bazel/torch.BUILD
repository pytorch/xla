package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob(
        ["torch/include/**/*.h"],
        ["torch/include/google/protobuf/**/*.h"],
    ),
    strip_include_prefix = "torch/include",
)

# Runtime headers, for importing <torch/torch.h>.
cc_library(
    name = "runtime_headers",
    hdrs = glob(["torch/include/torch/csrc/api/include/**/*.h"]),
    strip_include_prefix = "torch/include/torch/csrc/api/include",
)

filegroup(
    name = "torchgen_deps",
    srcs = [
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/tags.yaml",
        "aten/src/ATen/native/ts_native_functions.yaml",
        "aten/src/ATen/templates/DispatchKeyNativeFunctions.cpp",
        "aten/src/ATen/templates/DispatchKeyNativeFunctions.h",
        "aten/src/ATen/templates/LazyIr.h",
        "aten/src/ATen/templates/LazyNonNativeIr.h",
        "aten/src/ATen/templates/RegisterDispatchDefinitions.ini",
        "aten/src/ATen/templates/RegisterDispatchKey.cpp",
        "torch/csrc/lazy/core/shape_inference.h",
        "torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
    ],
)

cc_import(
    name = "libtorch",
    shared_library = "build/lib/libtorch.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "build/lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "build/lib/libtorch_python.so",
)

cc_import(
    name = "libc10",
    shared_library = "build/lib/libc10.so",
)
