load(
    "@org_tensorflow//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files([
    "tf_version_script.lds",
    "tf_exported_symbols.lds",
])


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
  deps = [
    ":computation_client",
    ":env_vars",
    ":pjrt_computation_client",
    "@org_tensorflow//tensorflow/tsl/platform:stacktrace",
  ],
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
    "@org_tensorflow//tensorflow/tsl/platform:stacktrace_handler",
    "@org_tensorflow//tensorflow/compiler/xla:literal_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:span",
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
    ":thread_pool",
    ":multi_wait",
    "@org_tensorflow//tensorflow/compiler/xla:literal",
    "@org_tensorflow//tensorflow/compiler/xla:shape_util",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
    "@org_tensorflow//tensorflow/compiler/xla/service:gpu_plugin",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:tpu_client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
    "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
    "@org_tensorflow//tensorflow/tsl/profiler/lib:traceme",
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
      "@org_tensorflow//tensorflow/tsl/platform:status",
      "@org_tensorflow//tensorflow/tsl/profiler/rpc:profiler_server_impl",
      "@org_tensorflow//tensorflow/tsl/profiler/rpc/client:capture_profile",
      "@com_google_absl//absl/container:flat_hash_map",

      # Profiler silently fails unless we include this
      "@org_tensorflow//tensorflow/compiler/xla/backends/profiler:profiler_backends",

      # TODO: We get missing symbol errors without these deps. Why aren't they
      # included transitively from TensorFlow/TSL?
      "@org_tensorflow//tensorflow/tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
      "@org_tensorflow//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc_impl",
      "@org_tensorflow//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc_impl",
      "@org_tensorflow//tensorflow/tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
      "@org_tensorflow//tensorflow/tsl/profiler/rpc/client:profiler_client",
    ],
)

cc_library(
    name = "stablehlo_helper",
    srcs = ["stablehlo_helper.cc"],
    hdrs = ["stablehlo_helper.h"],
    deps = [
        ":types",
        ":xla_util",
        "@org_tensorflow//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@stablehlo//:stablehlo_serialization",
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
        "@org_tensorflow//tensorflow/tsl/platform:hash",
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
    size = "small",
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

cc_binary(
    name = "libxla_computation_client.so",
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location :tf_version_script.lds)",
            "-Wl,-soname,libxla_computation_client.so",
        ],
    }),
    linkshared = 1,
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
