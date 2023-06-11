load("@tsl//tsl:tsl.bzl", "if_with_tpu_support")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

load(
    "@tsl//tsl/platform/default:build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_cc",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@tsl//tsl/platform/default:rules_cc.bzl",
    "cc_binary",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files([
    "openxla_version_script.lds",
    "openxla_exported_symbols.lds",
])

config_setting(
  name = "disable_xrt",
  define_values = {"disable_xrt": "true"},
)

tf_proto_library(
    name = "topology_proto",
    srcs = [
        "topology.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "mesh_service_proto",
    srcs = ["mesh_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        ":topology_proto",
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
    size = "small",
    srcs = ["async_task_test.cc"],
    deps = [
        ":async_task",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
  name = "runtime",
  srcs = [
    "runtime.cc",
  ],
  hdrs = [
    "runtime.h",
  ],
  local_defines = select({
    ":disable_xrt": ["DISABLE_XRT"],
    "//conditions:default": [],
  }),
  deps = [
    ":computation_client",
    ":env_vars",
    ":pjrt_computation_client",
    "@tsl//tsl/platform:stacktrace",
  ] + select({
    ":disable_xrt": [],
    "//conditions:default": [],
  }),
)

cc_library(
  name = "computation_client",
  srcs = [
    "computation_client.cc",
  ],
  hdrs = [
    "computation_client.h",
  ],
  deps = [
    ":debug_macros",
    ":env_vars",
    ":metrics_analysis",
    ":metrics_reader",
    ":metrics",
    ":sys_util",
    ":types",
    ":util",
    "@xla//xla:literal_util",
    "@xla//xla/client:xla_computation",
    "@xla//xla/stream_executor/gpu",
    "@xla//xla/stream_executor/gpu:gpu_executor_header",
    "@xla//xla/stream_executor/gpu:gpu_activation_header",
    "@xla//xla/stream_executor/gpu:gpu_executor_header",
    "@xla//xla/stream_executor/gpu:gpu_helpers_header",
    "@xla//xla/stream_executor/gpu:gpu_stream_header",
    "@xla//xla/stream_executor/gpu:gpu_timer_header",
    "@xla//xla/stream_executor/gpu:gpu_types_header",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:span",
    "@tsl//tsl/platform:stacktrace_handler",
    "@tsl//tsl/protobuf:autotuning_proto_cc",
    "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
    "@tsl//tsl/protobuf:dnn_proto_cc",
    "@tsl//tsl/protobuf:dnn_proto_cc_impl",
  ],
)

cc_library(
  name = "pjrt_computation_client",
  srcs = [
    "pjrt_computation_client.cc",
  ],
  hdrs = [
    "pjrt_computation_client.h",
  ],
  deps = [
    ":computation_client",
    ":debug_macros",
    ":env_vars",
    ":tf_logging",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla/client:xla_computation",
    "@xla//xla/pjrt/distributed",
    "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
    "@xla//xla/pjrt:tpu_client",
    "@xla//xla/pjrt:pjrt_client",
    "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
    "@xla//xla/pjrt:pjrt_c_api_client",
    "@tsl//tsl/profiler/lib:traceme",
    "@xla//xla/service:gpu_plugin",
    ":thread_pool",
    ":multi_wait",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:span",
  ],
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
)

cc_test(
    name = "cache_test",
    size = "small",
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
        "@xla//xla:statusor",
        "@tsl//tsl/platform:stacktrace",
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
        "@xla//xla:statusor",
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
        "@xla//xla:types",
        "@xla//xla/service:platform_util",
    ],
)

cc_library(
    name = "multi_wait",
    srcs = ["multi_wait.cc"],
    hdrs = ["multi_wait.h"],
    deps = [
        "@xla//xla:types",
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
        "@xla//xla:types",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]),
)

# cc_library(
#     name = "profiler",
#     srcs = ["profiler.cc"],
#     hdrs = ["profiler.h"],
#     deps = [
#       "@tsl//tsl/profiler/rpc:profiler_server_impl",
#       "@tsl//tsl/profiler/rpc/client:profiler_client",
#       "@tsl//tsl/platform:status",
#       "@tsl//tsl/profiler/rpc/client:capture_profile",
#       "@com_google_absl//absl/container:flat_hash_map",
#       "@xla//xla/backends/profiler:profiler_backends",
#       "@tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
#       "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
#       "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
#       "@tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
#     ],
# )

cc_library(
    name = "record_reader",
    srcs = ["record_reader.cc"],
    hdrs = ["record_reader.h"],
    deps = [
        ":debug_macros",
        "@xla//xla:types",
        "@tsl//tsl/lib/io:record_reader",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "stablehlo_helper",
    srcs = ["stablehlo_helper.cc"],
    hdrs = ["stablehlo_helper.h"],
    deps = [
        ":types",
        ":xla_util",
        "@xla//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:hlo_proto_util",
    ],
)

cc_library(
    name = "sys_util",
    srcs = ["sys_util.cc"],
    hdrs = ["sys_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla:types",
    ],
)

cc_test(
    name = "sys_util_test",
    size = "small",
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
        "@xla//xla:statusor",
        "@xla//xla/service:platform_util",
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
        "@xla//xla:types",
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
    name = "conv_op_helpers",
    srcs = ["conv_op_helpers.cc"],
    hdrs = ["conv_op_helpers.h"],
    deps = [
        "@xla//xla/client/lib:arithmetic",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal_util",
        "@xla//xla:statusor",
        "@xla//xla/client:xla_builder",
        "@xla//xla/stream_executor:dnn",
        "@xla//xla:xla_data_proto_cc",
        "@tsl//tsl/protobuf:autotuning_proto_cc",
        "@tsl//tsl/protobuf:dnn_proto_cc",
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
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla/stream_executor/gpu",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_activation_header",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_helpers_header",
        "@xla//xla/stream_executor/gpu:gpu_stream_header",
        "@xla//xla/stream_executor/gpu:gpu_timer_header",
        "@xla//xla/stream_executor/gpu:gpu_types_header",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/protobuf:autotuning_proto_cc",
        "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
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
        "@com_google_absl//absl/types:span",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:types",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:platform_util",
        "@xla//xla/service/spmd:spmd_partitioner",
        "@tsl//tsl/platform:errors",
    ],
)

cc_test(
    name = "xla_util_test",
    srcs = ["xla_util_test.cc"],
    deps = [
        ":xla_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/protobuf:error_codes_proto_impl",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_binary(
    name = "libxla_computation_client.so",
    linkopts = [
        "-z defs",
        "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
        "$(location :tf_version_script.lds)",
        "-Wl,-soname,libxla_computation_client.so",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":computation_client",
        ":openxla_exported_symbols.lds",
        ":openxla_version_script.lds",
        "@com_google_absl//absl/strings",
        "@xla//xla:literal_util",
        "@xla//xla/client",
        "@xla//xla/client:global_data",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "@xla//xla/client/lib:svd",
        "@xla//xla/stream_executor/gpu",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_activation_header",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_helpers_header",
        "@xla//xla/stream_executor/gpu:gpu_stream_header",
        "@xla//xla/stream_executor/gpu:gpu_timer_header",
        "@xla//xla/stream_executor/gpu:gpu_types_header",
        "@tsl//tsl/platform/cloud:gcs_file_system",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:protobuf_compiler",
        "@tsl//tsl/profiler/convert:xplane_to_trace_events",
        "@tsl//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "@tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
        "@tsl//tsl/profiler/rpc:profiler_server_for_pybind",
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
        "@tsl//tsl/profiler/rpc:profiler_service_impl",
        "@tsl//tsl/profiler/rpc/client:capture_profile",
        "@tsl//tsl/profiler/rpc/client:profiler_client_impl",
        "@tsl//tsl/profiler/rpc/client:save_profile",
        "@tsl//tsl/protobuf:bfc_memory_map_proto_cc_impl",
        "@tsl//tsl/protobuf:autotuning_proto_cc",
        "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@xla//xla:xla_data_cc_grpc_proto",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/client/lib:self_adjoint_eig",
        "@xla//xla/rpc:grpc_stub",
        "@xla//xla/rpc:xla_service_cc_grpc_proto",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

# TODO(goranpetrovic): reenable when `xla_cc_test` is fixed upstream.
# xla_cc_test(
#     name = "pjrt_computation_client_test",
#     srcs = ["pjrt_computation_client_test.cc"],
#     deps = [
#         ":computation_client",
#         "@xla//xla:literal",
#         "@xla//xla:literal_util",
#         "@xla//xla:shape_util",
#         "@xla//xla:status",
#         "@xla//xla:statusor",
#         "@xla//xla/client:xla_builder",
#         "@xla//xla/client:xla_computation",
#         "@xla//xla/tests:literal_test_util",
#         "@xla//xla/tools:hlo_module_loader",
#         "@org_tensorflow//tensorflow/core/platform:logging",
#         "@tsl//tsl/lib/core:status_test_util",
#         "@tsl//tsl/platform:env",
#         "@tsl//tsl/platform:test",
#         "@tsl//tsl/platform:test_main",
#     ],
# )
