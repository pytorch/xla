load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_test",
)

load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_test",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

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
    "@tsl//tsl/platform:stacktrace",
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
  copts = [
    "-isystemexternal/torch",
  ],
  deps = [
    ":debug_macros",
    ":env_vars",
    ":metrics_analysis",
    ":metrics_reader",
    ":metrics",
    ":sys_util",
    ":tensor_source",
    ":types",
    ":util",
    ":xla_coordinator",
    "//torch_xla/csrc:device",
    "//torch_xla/csrc:dtype",
    "@tsl//tsl/platform:stacktrace_handler",
    "@xla//xla:literal_util",
    "@xla//xla/client:xla_computation",
    "@xla//xla/hlo/ir:hlo",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:span",
    "@torch//:headers",
    "@torch//:runtime_headers",
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
    ":profiler",
    ":stablehlo_helper",
    ":tensor_source",
    ":tf_logging",
    ":xla_coordinator",
    "//torch_xla/csrc:thread_pool",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla/client:xla_computation",
    "@xla//xla/pjrt/distributed",
    "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
    "@xla//xla/service:gpu_plugin",
    "@xla//xla/pjrt:pjrt_client",
    "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
    "@xla//xla/pjrt:pjrt_c_api_client",
    "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
    "@tsl//tsl/profiler/lib:traceme",
    "@tsl//tsl/platform/cloud:gcs_file_system",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/synchronization",
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
    name = "xla_coordinator",
    srcs = ["xla_coordinator.cc"],
    hdrs = ["xla_coordinator.h"],
    deps = [
        ":debug_macros",
        ":sys_util",
        "@xla//xla/pjrt/distributed",
        "@tsl//tsl/distributed_runtime/preemption:preemption_sync_manager",
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

# Profiler silently fails unless we link these backends
cc_library(
  name = "profiler_backends",
  visibility = ["//visibility:private"],
  deps = [
    "@xla//xla/backends/profiler/cpu:host_tracer",
    "@xla//xla/backends/profiler/cpu:metadata_collector",
  ] + if_cuda_is_configured([
    "@xla//xla/backends/profiler/gpu:device_tracer",
  ]),
  alwayslink = True,
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
      ":debug_macros",
      ":profiler_backends",
      "@xla//xla/backends/profiler/plugin:profiler_c_api_hdrs",
      "@xla//xla/backends/profiler/plugin:plugin_tracer",
      "@xla//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
      "@tsl//tsl/platform:status",
      "@tsl//tsl/profiler/lib:profiler_factory",
      "@tsl//tsl/profiler/rpc:profiler_server_impl",
      "@tsl//tsl/profiler/rpc/client:capture_profile",
      "@com_google_absl//absl/container:flat_hash_map",

      # TODO: We get missing symbol errors without these deps. Why aren't they
      # included transitively from TensorFlow/TSL?
      "@tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
      "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
      "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
      "@tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
      "@tsl//tsl/profiler/rpc/client:profiler_client",
    ],
)

cc_library(
    name = "stablehlo_helper",
    srcs = ["stablehlo_helper.cc"],
    hdrs = ["stablehlo_helper.h"],
    deps = [
        ":types",
        ":xla_util",
        "@stablehlo//:stablehlo_serialization",
        "@xla//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/mlir_hlo:all_passes",
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
  name = "tensor_source",
  hdrs = ["tensor_source.h"],
  deps = [
      ":debug_macros",
      "@xla//xla:literal",
      "@xla//xla:shape_util",
      "@torch//:headers",
  ]
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
    name = "util",
    hdrs = ["util.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@tsl//tsl/platform:hash",
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
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:platform_util",
        "@xla//xla/service/spmd:spmd_partitioner",
        "@tsl//tsl/platform:errors",
        "@torch//:headers",
    ],
)

ptxla_cc_test(
    name = "xla_util_test",
    size = "small",
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
        "@tsl//tsl/platform:status_matchers",
    ],
)

ptxla_cc_test(
    name = "pjrt_computation_client_test",
    srcs = ["pjrt_computation_client_test.cc"],
    deps = [
        ":computation_client",
        ":pjrt_computation_client",
        ":tensor_source",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "@xla//xla/tests:literal_test_util",
        "@xla//xla/tools:hlo_module_loader",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ],
)
