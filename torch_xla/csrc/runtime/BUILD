load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_test",
)
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "runtime",
    srcs = [
        "runtime.cpp",
    ],
    hdrs = [
        "runtime.h",
    ],
    deps = [
        ":computation_client",
        ":env_vars",
        ":ifrt_computation_client",
        ":pjrt_computation_client",
        "@tsl//tsl/platform:stacktrace",
    ],
)

cc_library(
    name = "computation_client",
    srcs = [
        "computation_client.cpp",
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
        ":metrics",
        ":metrics_analysis",
        ":metrics_reader",
        ":sys_util",
        ":tensor_source",
        ":types",
        ":util",
        ":xla_coordinator",
        ":xla_util",
        "//torch_xla/csrc:device",
        "//torch_xla/csrc:dtype",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@torch//:headers",
        "@torch//:runtime_headers",
        "@tsl//tsl/platform:stacktrace_handler",
        "@xla//xla:literal_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "ifrt_computation_client",
    srcs = [
        "ifrt_computation_client.cpp",
    ],
    hdrs = [
        "ifrt_computation_client.h",
    ],
    deps = [
        ":computation_client",
        ":debug_macros",
        ":env_vars",
        ":operation_manager",
        ":pjrt_registry",
        ":stablehlo_helper",
        ":tf_logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/distributed",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "@xla//xla/python/ifrt:attribute_map",
    ],
)

cc_library(
    name = "pjrt_computation_client",
    srcs = [
        "pjrt_computation_client.cpp",
    ],
    hdrs = [
        "pjrt_computation_client.h",
    ],
    deps = [
        ":computation_client",
        ":debug_macros",
        ":env_hash",
        ":env_vars",
        ":operation_manager",
        ":pjrt_registry",
        ":profiler",
        ":stablehlo_helper",
        ":tensor_source",
        ":tf_logging",
        ":xla_coordinator",
        "//torch_xla/csrc:thread_pool",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt/distributed",
        "@xla//xla/service:custom_call_target_registry",
    ],
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
    deps = [
        "@torch//:headers",
    ],
)

cc_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.cpp"],
    deps = [
        ":cache",
        "@com_google_googletest//:gtest_main",
        "@torch//:libtorch_cpu",  # For TORCH_LAZY_COUNTER
    ],
)

cc_library(
    name = "debug_macros",
    hdrs = ["debug_macros.h"],
    deps = [
        ":tf_logging",
        "@tsl//tsl/platform:stacktrace",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:macros",
    ],
)

cc_library(
    name = "env_vars",
    srcs = ["env_vars.cpp"],
    hdrs = ["env_vars.h"],
)

cc_library(
    name = "env_hash",
    srcs = ["env_hash.cpp"],
    hdrs = ["env_hash.h"],
    deps = [
        ":sys_util",
        "@torch//:headers",
    ],
)

cc_test(
    name = "env_hash_test",
    size = "small",
    srcs = ["env_hash_test.cpp"],
    deps = [
        ":env_hash",
        "@com_google_googletest//:gtest_main",
        "@torch//:libtorch_cpu",  # For torch::lazy::hash
    ],
)

cc_library(
    name = "pjrt_registry",
    srcs = ["pjrt_registry.cpp"],
    hdrs = ["pjrt_registry.h"],
    deps = [
        ":debug_macros",
        ":env_hash",
        ":env_vars",
        ":profiler",
        ":sys_util",
        ":tf_logging",
        ":xla_coordinator",
        "@com_google_absl//absl/log:initialize",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/service:gpu_plugin",
    ],
)

cc_library(
    name = "metrics_analysis",
    srcs = ["metrics_analysis.cpp"],
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
    srcs = ["metrics_reader.cpp"],
    hdrs = ["metrics_reader.h"],
    deps = [
        ":debug_macros",
        ":metrics",
        ":util",
    ],
)

cc_library(
    name = "xla_coordinator",
    srcs = ["xla_coordinator.cpp"],
    hdrs = ["xla_coordinator.h"],
    deps = [
        ":debug_macros",
        ":sys_util",
        ":env_vars",
        "@xla//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "@xla//xla/pjrt/distributed",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cpp"],
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
    name = "operation_manager",
    srcs = ["operation_manager.cpp"],
    hdrs = ["operation_manager.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":debug_macros",
        ":tf_logging",
        "@com_google_absl//absl/types:span",
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
    srcs = ["profiler.cpp"],
    hdrs = ["profiler.h"],
    deps = [
        ":tf_logging",
        ":profiler_backends",
        "@com_google_absl//absl/status",
        "@xla//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "@xla//xla/backends/profiler/plugin:plugin_tracer",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@com_google_absl//absl/container:flat_hash_map",

        # TODO: We get missing symbol errors without these deps. Why aren't they
        # included transitively from TensorFlow/TSL?
        "@tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client",
    ],
)

cc_library(
    name = "stablehlo_composite_helper",
    srcs = ["stablehlo_composite_helper.cpp"],
    hdrs = ["stablehlo_composite_helper.h"],
    deps = [
        ":types",
        ":xla_util",
        "@com_nlohmann_json//:json",
        "@xla//xla/mlir_hlo:all_passes",
    ],
)

cc_library(
    name = "xla_mlir_debuginfo_helper",
    srcs = ["xla_mlir_debuginfo_helper.cpp"],
    hdrs = ["xla_mlir_debuginfo_helper.h"],
    deps = [
        ":types",
        ":xla_util",
        "@xla//xla/mlir_hlo:all_passes",
    ],
)

cc_library(
    name = "stablehlo_helper",
    srcs = ["stablehlo_helper.cpp"],
    hdrs = ["stablehlo_helper.h"],
    deps = [
        ":stablehlo_composite_helper",
        ":types",
        ":xla_mlir_debuginfo_helper",
        ":xla_util",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_serialization",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@xla//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
    ],
)

cc_library(
    name = "sys_util",
    srcs = ["sys_util.cpp"],
    hdrs = ["sys_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla:types",
    ],
)

cc_test(
    name = "sys_util_test",
    size = "small",
    srcs = ["sys_util_test.cpp"],
    deps = [
        ":sys_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_logging",
    srcs = ["tf_logging.cpp"],
    hdrs = ["tf_logging.h"],
    deps = [
        "@tsl//tsl/platform:statusor",
        "@xla//xla/service:platform_util",
        "@com_google_absl//absl/base:log_severity",
    ],
)

cc_library(
    name = "tensor_source",
    hdrs = ["tensor_source.h"],
    deps = [
        ":debug_macros",
        "@torch//:headers",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
    ],
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
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:types",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cpp"],
    deps = [
        ":util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "xla_util",
    srcs = ["xla_util.cpp"],
    hdrs = ["xla_util.h"],
    deps = [
        ":metrics",
        ":sys_util",
        ":tf_logging",
        ":types",
        ":util",
        "@com_google_absl//absl/types:span",
        "@torch//:headers",
        "@tsl//tsl/platform:errors",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:types",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:platform_util",
        "@xla//xla/service/spmd:spmd_partitioner",
    ],
)

ptxla_cc_test(
    name = "xla_util_test",
    size = "small",
    srcs = ["xla_util_test.cpp"],
    deps = [
        ":debug_macros",
        ":xla_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status_matchers",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
    ],
)

ptxla_cc_test(
    name = "pjrt_computation_client_test",
    srcs = ["pjrt_computation_client_test.cpp"],
    deps = [
        ":computation_client",
        ":pjrt_computation_client",
        ":tensor_source",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/tests:literal_test_util",
        "@xla//xla/tools:hlo_module_loader",
    ],
)

# ptxla_cc_test(
#     name = "ifrt_computation_client_test",
#     srcs = ["ifrt_computation_client_test.cpp"],
#     deps = [
#         ":computation_client",
#         ":ifrt_computation_client",
#         ":tensor_source",
#         "@xla//xla/tsl/lib/core:status_test_util",
#         "@tsl//tsl/platform:env",
#         "@tsl//tsl/platform:errors",
#         "@tsl//tsl/platform:logging",
#         "@tsl//tsl/platform:test",
#         "@tsl//tsl/platform:test_main",
#         "@xla//xla:literal",
#         "@xla//xla:literal_util",
#         "@xla//xla:shape_util",
#         "@xla//xla:status",
#         "@xla//xla:statusor",
#         "@xla//xla/hlo/builder:xla_builder",
#         "@xla//xla/hlo/builder:xla_computation",
#         "@xla//xla/tests:literal_test_util",
#         "@xla//xla/tools:hlo_module_loader",
#     ],
# )
