load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

cc_binary(
    name = "_XLAC.so",
    copts = [
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_XLAC",
        "-fopenmp",
        "-fPIC",
        "-fwrapv",
    ],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/torch_xla/lib",  # for libtpu
        "-Wl,-soname,_XLAC.so",
        "-lstdc++fs",  # For std::filesystem
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//torch_xla/csrc:init_python_bindings",
        "@torch//:headers",
        "@torch//:libc10",
        "@torch//:libtorch",
        "@torch//:libtorch_cpu",
        "@torch//:libtorch_python",
    ] + if_cuda_is_configured([
        "@xla//xla/stream_executor:cuda_platform",
    ]),
)
