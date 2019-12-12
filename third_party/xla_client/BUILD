licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)
load(
    "//tensorflow/core/platform:default/cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
    ],
)

tf_proto_library_cc(
    name = "mesh_service_proto",
    srcs = ["mesh_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//tensorflow/core/protobuf/tpu:topology_proto",
    ],
)

tf_cc_shared_object(
    name = "libxla_computation_client.so",
    linkopts = select({
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
        "//tensorflow/compiler/xla/client/lib:svd",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/compiler/xla/xla_client:tf_exported_symbols.lds",
        "//tensorflow/compiler/xla/xla_client:tf_version_script.lds",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform/cloud:gcs_file_system",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "computation_client_impl",
    srcs = [
        "computation_client.cc",
        "mesh_service.cc",
        "metrics.cc",
        "multi_wait.cc",
        "record_reader.cc",
        "sys_util.cc",
        "tf_logging.cc",
        "thread_pool.cc",
        "triggered_task.cc",
        "xla_util.cc",
        "xrt_computation_client.cc",
        "xrt_local_service.cc",
        "xrt_session.cc",
        "xrt_session_cache.cc",
    ],
    hdrs = [
        "cache.h",
        "computation_client.h",
        "debug_macros.h",
        "mesh_service.h",
        "metrics.h",
        "multi_wait.h",
        "record_reader.h",
        "sys_util.h",
        "tf_logging.h",
        "thread_pool.h",
        "triggered_task.h",
        "unique.h",
        "util.h",
        "xla_util.h",
        "xrt_computation_client.h",
        "xrt_local_service.h",
        "xrt_session.h",
        "xrt_session_cache.h",
    ],
    deps = [
        ":mesh_service_proto_cc",
        "//tensorflow:grpc",
        "//tensorflow:grpc++",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/stream_executor:stream_executor_impl",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured(["//tensorflow/compiler/jit:xla_gpu_device"]),
    alwayslink = 1,
)
