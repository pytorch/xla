"""PyTorch sources and libraries provider."""

package(
    default_visibility = [
        "//visibility:public",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "**/*.h",
        "**/*.cpp",
        "**/*.yaml",
        "**/*.ini",
    ]),
)

cc_library(
    name = "headers",
    srcs = glob(
        [
            "torch/include/**/*.h",
            "torch/csrc/lazy/python/*.h",
        ],
        ["torch/include/google/protobuf/**/*.h"],
    ),
)

cc_import(
    name = "libtorch",
    hdrs = [":headers"],
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
