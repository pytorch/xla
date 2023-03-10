load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
)
load(
    "@org_tensorflow//tensorflow/tsl/platform/default:build_config.bzl",
    "tf_proto_library_cc",
)
load(
    "@org_tensorflow//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//bazel:tensorflow.bzl",
    "if_with_tpu_support",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files([
    "tf_version_script.lds",
    "tf_exported_symbols.lds",
])

tf_proto_library_cc(
    name = "mesh_service_proto",
    srcs = ["mesh_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto",
    ],
)

cc_library(
    name = "async_task",
    hdrs = ["async_task.h"],
    deps = [
        ":debug_macros",
        ":thread_pool",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "async_task_test",
    srcs = ["async_task_test.cc"],
    deps = [
        ":async_task",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "computation_client",
    srcs = [
        "computation_client.cc",
        "pjrt_computation_client.cc",
        "xrt_computation_client.cc",
    ],
    hdrs = [
        "computation_client.h",
        "pjrt_computation_client.h",
        "xrt_computation_client.h",
    ],
    deps = [
        ":cache",
        ":debug_macros",
        ":env_vars",
        ":mesh_service",
        ":metrics_analysis",
        ":metrics_reader",
        ":metrics",
        ":multi_wait",
        ":profiler",
        ":record_reader",
        ":sys_util",
        ":tf_logging",
        ":thread_pool",
        ":triggered_task",
        ":types",
        ":unique",
        ":util",
        ":xla_util",
        ":xrt_local_service",
        ":xrt_session",
        ":xrt_session_cache",
        "@org_tensorflow//tensorflow:grpc++",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla:debug_options_flags",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/client",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:arithmetic",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:comparators",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:constants",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:logdet",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:math",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:matrix",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:pooling",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:qr",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:slicing",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:sorting",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:svd",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:tridiagonal",
        "@org_tensorflow//tensorflow/compiler/xla/client:global_data",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:tpu_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
        "@org_tensorflow//tensorflow/compiler/xla/rpc:grpc_stub",
        "@org_tensorflow//tensorflow/compiler/xla/service:cpu_plugin",
        "@org_tensorflow//tensorflow/compiler/xla/service:platform_util",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/hlo/ir:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
        "@org_tensorflow//tensorflow/compiler/xrt:xrt_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xrt:xrt_server",
        "@org_tensorflow//tensorflow/compiler/xrt:xrt_utils",
        "@org_tensorflow//tensorflow/compiler/xrt/cc:xrt_ops",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/distributed_runtime:server_lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "@org_tensorflow//tensorflow/core/kernels:data_flow",
        "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/rpc:profiler_server_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/rpc/client:profiler_client",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor:cuda_platform",
    ]) + if_with_tpu_support([
        "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_device",
        "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_jit",
    ]),
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
)

cc_test(
    name = "cache_test",
    srcs = ["cache_test.cc"],
    deps = [
        ":cache",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "debug_macros",
    hdrs = ["debug_macros.h"],
    deps = [
        ":tf_logging",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/tsl/platform:stacktrace",
    ],
)

cc_library(
    name = "env_vars",
    srcs = ["env_vars.cc"],
    hdrs = ["env_vars.h"],
)

cc_library(
    name = "mesh_service",
    srcs = ["mesh_service.cc"],
    hdrs = ["mesh_service.h"],
    deps = [
        "nccl_distributed",
        ":debug_macros",
        ":mesh_service_proto_cc",
        ":multi_wait",
        ":sys_util",
        ":thread_pool",
        ":util",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "metrics_analysis",
    srcs = ["metrics_analysis.cc"],
    hdrs = ["metrics_analysis.h"],
    deps = [
        ":metrics",
        ":tf_logging",
        ":types",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "metrics_reader",
    srcs = ["metrics_reader.cc"],
    hdrs = ["metrics_reader.h"],
    deps = [
        ":debug_macros",
        ":metrics",
        ":util",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        ":debug_macros",
        ":sys_util",
        ":util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla/service:platform_util",
    ],
)

cc_library(
    name = "multi_wait",
    srcs = ["multi_wait.cc"],
    hdrs = ["multi_wait.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "nccl_distributed",
    srcs = ["nccl_distributed.cc"],
    hdrs = ["nccl_distributed.h"],
    deps = [
        ":debug_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]),
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        "@org_tensorflow//tensorflow/tsl/profiler/rpc:profiler_server_impl",
    ],
)

cc_library(
    name = "record_reader",
    srcs = ["record_reader.cc"],
    hdrs = ["record_reader.h"],
    deps = [
        ":debug_macros",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/tsl/lib/io:record_reader",
        "@org_tensorflow//tensorflow/tsl/platform:errors",
        "@org_tensorflow//tensorflow/tsl/platform:strcat",
    ],
)

cc_library(
    name = "sys_util",
    srcs = ["sys_util.cc"],
    hdrs = ["sys_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

cc_test(
    name = "sys_util_test",
    srcs = ["sys_util_test.cc"],
    deps = [
        ":sys_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_logging",
    srcs = ["tf_logging.cc"],
    hdrs = ["tf_logging.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/service:platform_util",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        ":metrics",
        ":tf_logging",
    ],
)

cc_library(
    name = "triggered_task",
    srcs = ["triggered_task.cc"],
    hdrs = ["triggered_task.h"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "unique",
    hdrs = ["unique.h"],
    deps = [
        ":debug_macros",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_util",
    srcs = ["xla_util.cc"],
    hdrs = ["xla_util.h"],
    deps = [
        ":metrics",
        ":sys_util",
        ":tf_logging",
        ":types",
        ":util",
        ":xrt_session",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status_macros",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:platform_util",
        "@org_tensorflow//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
        "@org_tensorflow//tensorflow/tsl/platform:errors",
    ],
)

cc_test(
    name = "xla_util_test",
    srcs = ["xla_util_test.cc"],
    deps = [
        ":xla_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
        "@org_tensorflow//tensorflow/tsl/lib/core:status_test_util",
        "@org_tensorflow//tensorflow/tsl/platform:errors",
        "@org_tensorflow//tensorflow/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "xrt_local_service",
    srcs = ["xrt_local_service.cc"],
    hdrs = ["xrt_local_service.h"],
    deps = [
        ":debug_macros",
        ":xrt_session",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor/tpu:tpu_initializer_helper",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime:server_lib",
        "@org_tensorflow//tensorflow/tsl/platform:errors",
        "@org_tensorflow//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "xrt_session_cache",
    srcs = ["xrt_session_cache.cc"],
    hdrs = ["xrt_session_cache.h"],
    deps = [
        ":metrics",
        ":sys_util",
        ":xrt_session",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "xrt_session",
    srcs = ["xrt_session.cc"],
    hdrs = ["xrt_session.h"],
    deps = [
        ":debug_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

tf_cc_shared_object(
    name = "libxla_computation_client.so",
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location :tf_version_script.lds)",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":computation_client",
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla/client",
        "@org_tensorflow//tensorflow/compiler/xla/client:global_data",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:svd",
        "@org_tensorflow//tensorflow/compiler/xla/rpc:grpc_stub",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/python/profiler/internal:profiler_pywrap_impl",
        "@org_tensorflow//tensorflow/tsl/platform/cloud:gcs_file_system",
    ],
)

# TODO(goranpetrovic): reenable when `xla_cc_test` is fixed upstream.
# xla_cc_test(
#     name = "pjrt_computation_client_test",
#     srcs = ["pjrt_computation_client_test.cc"],
#     deps = [
#         ":computation_client",
#         "@org_tensorflow//tensorflow/compiler/xla:literal",
#         "@org_tensorflow//tensorflow/compiler/xla:literal_util",
#         "@org_tensorflow//tensorflow/compiler/xla:shape_util",
#         "@org_tensorflow//tensorflow/compiler/xla:status",
#         "@org_tensorflow//tensorflow/compiler/xla:statusor",
#         "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
#         "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
#         "@org_tensorflow//tensorflow/compiler/xla/tests:literal_test_util",
#         "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
#         "@org_tensorflow//tensorflow/core/platform:logging",
#         "@org_tensorflow//tensorflow/tsl/lib/core:status_test_util",
#         "@org_tensorflow//tensorflow/tsl/platform:env",
#         "@org_tensorflow//tensorflow/tsl/platform:test",
#         "@org_tensorflow//tensorflow/tsl/platform:test_main",
#     ],
# )
