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
    "openxla_version_script.lds", # this file has been added into third_party/xla_client
    "openxla_exported_symbols.lds", # this file has been added into third_party/xla_client
])

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

tf_proto_library(
    name = "example_protos",
    srcs = [
        "example.proto",
        "feature.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "async_task",
    hdrs = ["async_task.h"],
    deps = [
        ":debug_macros",
        ":thread_pool",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = True,
)

cc_library(
    name = "computation_client",
    srcs = [
        "computation_client.cc",
        "pjrt_computation_client.cc",
	],
    hdrs = [
        "computation_client.h",
        "pjrt_computation_client.h",
    ],
    deps = [
        ":cache",
        ":debug_macros",
        ":env_vars",
        ":example_protos_cc",
        ":mesh_service", # TODO(manfei@): update related to keep this
        ":metrics_analysis",
        ":metrics_reader",
        ":metrics",
        ":multi_wait",
        ":profiler", # TODO(manfei@): update related to keep this or delete
        ":record_reader",
        ":sys_util",
        ":tf_logging",
        ":thread_pool",
        ":triggered_task",
        ":types",
        ":unique",
        ":util",
        ":xla_util",
        "@com_github_grpc_grpc//:grpc++",
        # "@org_tensorflow//tensorflow/cc:client_session",
        # "@org_tensorflow//tensorflow/cc:scope",
        "@xla//xla:debug_options_flags",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client/lib:arithmetic",
        "@xla//xla/client/lib:comparators",
        "@xla//xla/client/lib:constants",
        "@xla//xla/client/lib:logdet",
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:matrix",
        "@xla//xla/client/lib:pooling",
        "@xla//xla/client/lib:qr",
        "@xla//xla/client/lib:slicing",
        "@xla//xla/client/lib:sorting",
        "@xla//xla/client/lib:svd",
        "@xla//xla/client/lib:tridiagonal",
        "@xla//xla/client:global_data",
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt/distributed",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/pjrt:tpu_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/rpc:grpc_stub",
        "@xla//xla/service:cpu_plugin",
        "@xla//xla/service:platform_util",
        "@xla//xla:statusor",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service/spmd:spmd_partitioner",
        # "@org_tensorflow//tensorflow/core:core_cpu",
        # "@org_tensorflow//tensorflow/core:framework_internal",
        # "@org_tensorflow//tensorflow/core:lib",
        # "@org_tensorflow//tensorflow/core:protos_all_cc",
        # "@org_tensorflow//tensorflow/core/distributed_runtime:server_lib",
        # "@tsl//tsl/profiler/protobuf/profiler_service_cc_impl",
        # "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        # "@org_tensorflow//tensorflow/core/kernels:data_flow",
        # "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@tsl//tsl/protobuf:coordination_config_proto_cc_impl",
        "@tsl//tsl/protobuf:distributed_runtime_payloads_proto_cc_impl",
        "@tsl//tsl/protobuf:histogram_proto_cc_impl",
        "@xla//xla/service/gpu:backend_configs_cc_impl",
        "@tsl//tsl/platform:net",
        "@tsl//tsl/platform:stacktrace_handler",
        "@tsl//tsl/util:device_name_utils",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
        "@tsl//tsl/profiler/rpc/client:profiler_client",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        # "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_device",
        "@xla//xla/stream_executor:cuda_platform",
    ]) + if_with_tpu_support([
        # "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_device",
        # "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_jit",
    ]),
    alwayslink = True,
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
    alwayslink = True,
)

cc_library(
    name = "debug_macros",
    hdrs = ["debug_macros.h"],
    deps = [
        ":tf_logging",
        "@xla//xla:statusor",
        "@tsl//tsl/platform:stacktrace",
    ],
    alwayslink = True,
)

cc_library(
    name = "env_vars",
    srcs = ["env_vars.cc"],
    hdrs = ["env_vars.h"],
    alwayslink = True,
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
    alwayslink = True,
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
    alwayslink = True,
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
    alwayslink = True,
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
    alwayslink = True,
)

cc_library(
    name = "multi_wait",
    srcs = ["multi_wait.cc"],
    hdrs = ["multi_wait.h"],
    deps = [
        "@xla//xla:types",
    ],
    alwayslink = True,
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
    alwayslink = True,
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
    ],
    alwayslink = True,
)

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
    alwayslink = True,
)

cc_library(
    name = "sys_util",
    srcs = ["sys_util.cc"],
    hdrs = ["sys_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla:types",
    ],
    alwayslink = True,
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
        "@xla//xla:statusor",
        "@xla//xla/service:platform_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        ":metrics",
        ":tf_logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "triggered_task",
    srcs = ["triggered_task.cc"],
    hdrs = ["triggered_task.h"],
    alwayslink = True,
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/types:optional",
        "@xla//xla:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "unique",
    hdrs = ["unique.h"],
    deps = [
        ":debug_macros",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = True,
)

cc_library(
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    deps = [
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/client:xla_builder",
    ],
)

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":constants",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:types",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
    ],
)

cc_library(
    name = "conv_op_helpers",
    srcs = ["conv_op_helpers.cc"],
    hdrs = ["conv_op_helpers.h"],
    deps = [
        ":arithmetic",
        ":constants",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal_util",
        "@xla//xla:statusor",
        "@xla//xla/client:xla_builder",
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
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:hash",
        # "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = True,
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
        "@com_google_absl//absl/types:span",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:types",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:platform_util",
        "@xla//xla/service/spmd:spmd_partitioner",
        "@tsl//tsl/platform:errors",
    ],
    alwayslink = True,
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
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_shared_library(
    name = "xla_computation_client",
    roots = [
	":computation_client",
    ],
    shared_lib_name = None,
    static_deps = [
	"//:__subpackages__",
        "@tsl//:__subpackages__",
        "@xla//:__subpackages__",
        "@tf_runtime//:__subpackages__",
        "@curl//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@stablehlo//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@mkl_dnn_v1//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@snappy//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@nsync//:__subpackages__",        "@double_conversion//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@upb//:__subpackages__",
        "@zlib//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@nccl_archive//:__subpackages__",
        "@cudnn_frontend_archive//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        # ":computation_client",
        # ":openxla_exported_symbols.lds",
        # ":openxla_version_script.lds",
        # "@com_google_absl//absl/strings",
        # "@xla//xla:literal_util",
        # "@xla//xla/client",
        # "@xla//xla/client:global_data",
        # "@xla//xla/client:xla_builder",
        # "@xla//xla/client:xla_computation",
        # "@xla//xla/client/lib:svd",
        # "@xla//xla/rpc:grpc_stub",
        # "@tsl//tsl/platform/cloud:gcs_file_system",
        # "@tsl//tsl/profiler/convert:xplane_to_trace_events",
    ],
)

# TODO(goranpetrovic): reenable when `xla_cc_test` is fixed upstream.
# xla_cc_test(
#     name = "pjrt_computation_client_test",
#     srcs = ["pjrt_computation_client_test.cc"],
#     deps = [
#         ":computation_client",
#         "@xla/xla:literal",
#         "@xla/xla:literal_util",
#         "@xla/xla:shape_util",
#         "@xla/xla:status",
#         "@xla/xla:statusor",
#         "@xla/xla/client:xla_builder",
#         "@xla/xla/client:xla_computation",
#         "@xla/xla/tests:literal_test_util",
#         "@xla/xla/tools:hlo_module_loader",
#         "@org_tensorflow//tensorflow/core/platform:logging",
#         "@tsl//tsl/lib/core:status_test_util",
#         "@tsl//tsl/platform:env",
#         "@tsl//tsl/platform:test",
#         "@tsl//tsl/platform:test_main",
#     ],
# )