licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
    ],
)

tf_cc_shared_object(
    name = "libxla_computation_client.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "$(location //tensorflow/compiler/xla/xla_client:tf_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow/compiler/xla/xla_client:tf_version_script.lds)",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "computation_client_impl",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/compiler/xla/xla_client:tf_exported_symbols.lds",
        "//tensorflow/compiler/xla/xla_client:tf_version_script.lds",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "computation_client_impl",
    srcs = [
        "computation_client.cc",
        "metrics.cc",
        "multi_wait.cc",
        "sys_util.cc",
        "thread_pool.cc",
        "xla_computation_client.cc",
        "xla_util.cc",
        "xrt_computation_client.cc",
        "xrt_session.cc",
        "xrt_session_cache.cc",
    ],
    hdrs = [
        "computation_client.h",
        "debug_macros.h",
        "metrics.h",
        "multi_wait.h",
        "sys_util.h",
        "thread_pool.h",
        "unique.h",
        "xla_computation_client.h",
        "xla_util.h",
        "xrt_computation_client.h",
        "xrt_session.h",
        "xrt_session_cache.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xrt:xrt_proto",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/contrib/tpu:all_ops",
        "//tensorflow/contrib/tpu/proto:topology_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/stream_executor:stream_executor_impl",
        "@com_google_absl//absl/strings",
    ],
)
