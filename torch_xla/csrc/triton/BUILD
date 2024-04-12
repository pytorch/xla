load("//bazel:rules_def.bzl","cc_proto_library",)

cc_library(
    name = "cuda_vendor",
    hdrs = [
        "gpu_vendor.h",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ],
)

proto_library(
    name = "triton_proto",
    srcs = ["triton.proto"],
)

cc_proto_library(
    name = "triton_cc_proto",
    deps = [":triton_proto"],
)

cc_library(
    name = "cuda_gpu_kernel_helpers",
    srcs = [
        "gpu_kernel_helpers.cpp",
    ],
    hdrs = [
        "gpu_kernel_helpers.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cuda_vendor",
        "@xla//xla/tsl/cuda:cupti",
        "@xla//xla/tsl/cuda:cusolver",
        "@xla//xla/tsl/cuda:cusparse",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "triton_utils",
    srcs = ["triton_utils.cpp"],
    hdrs = ["triton_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":triton_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@zlib",
    ],
)

cc_library(
    name = "triton_kernels",
    srcs = ["triton_kernels.cpp"],
    hdrs = ["triton_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":triton_utils",
        ":triton_cc_proto",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/gpu:asm_compiler",
        "@xla//xla/tsl/cuda:cudart",
        "@tsl//tsl/platform:env",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "kernel_nanobind_helpers",
    hdrs = ["kernel_nanobind_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@nanobind",
    ],
)

cc_library(
    name = "absl_status_casters",
    hdrs = ["absl_status_casters.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "_triton.so",
    srcs = ["triton.cc"],
    copts = [
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_triton",
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules", "-parse_headers",],
    linkopts = select({
        "//conditions:default": ["-Wl,-Bsymbolic","-Wl,-soname,_triton.so",],
    }),
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":triton_kernels",
        ":triton_utils",
        ":absl_status_casters",
        ":kernel_nanobind_helpers",
        ":triton_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)