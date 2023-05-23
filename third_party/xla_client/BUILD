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

# openxla_common : used
cc_library(
    name = "openxla_common",
    srcs = [
        "openxla_defs.cc",
    ],
    hdrs = [
        "openxla_defs.h",
    ],
    visibility = [":friends"],
)

# # openxla_flags
# cc_library(
#     name = "openxla_flags",
#     srcs = ["openxla_flags.cc"],
#     hdrs = ["openxla_flags.h"],
#     visibility = [":friends"],
#     deps = [
#         "//tensorflow/compiler/mlir/tensorflow:dump_graph",
#         "@xla//xla:parse_flags_from_env",
#         "//tensorflow/core:framework_internal",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:protos_all_cc",
#         "@com_google_absl//absl/base",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#     ],
# )

# # openxla_compilation_passes
# cc_library(
#     name = "openxla_compilation_passes",
#     srcs = [
#         "build_xla_ops_pass.cc",
#         "clone_constants_for_better_clustering.cc",
#         "cluster_scoping_pass.cc",
#         "deadness_analysis.cc",
#         "deadness_analysis_internal.h",
#         "encapsulate_subgraphs_pass.cc",
#         "encapsulate_xla_computations_pass.cc",
#         "extract_outside_compilation_pass.cc",
#         "force_xla_constants_on_host_pass.cc",
#         "increase_dynamism_for_auto_jit_pass.cc",
#         "mark_for_compilation_pass.cc",
#         "mark_for_compilation_pass_test_helper.cc",
#         "partially_decluster_pass.cc",
#         "report_clustering_info_pass.cc",
#     ],
#     hdrs = [
#         "build_xla_ops_pass.h",
#         "clone_constants_for_better_clustering.h",
#         "cluster_scoping_pass.h",
#         "deadness_analysis.h",
#         "encapsulate_subgraphs_pass.h",
#         "encapsulate_xla_computations_pass.h",
#         "extract_outside_compilation_pass.h",
#         "force_xla_constants_on_host_pass.h",
#         "increase_dynamism_for_auto_jit_pass.h",
#         "mark_for_compilation_pass.h",
#         "mark_for_compilation_pass_test_helper.h",
#         "partially_decluster_pass.h",
#         "report_clustering_info_pass.h",
#     ],
#     visibility = [
#         ":internal",
#         "//tensorflow/core/tfrt/utils:__pkg__",
#         "//third_party/cloud_tpu/inference_converter:__pkg__",
#     ],
#     deps = [
#         "compilability_check_util",
#         ":openxla_common",
#         ":device_util",
#         ":encapsulate_util",
#         ":openxla_flags",
#         ":resource_operation_safety_analysis",
#         ":shape_inference_helpers",
#         ":xla_activity_listener",
#         ":xla_cluster_util",
#         "//tensorflow/cc:cc_ops",
#         "//tensorflow/cc:functional_ops",
#         "//tensorflow/cc:ops",
#         "//tensorflow/cc:scope",
#         "//tensorflow/cc:scope_internal",
#         "//tensorflow/compiler/jit/ops:xla_ops",
#         "//tensorflow/compiler/tf2xla:resource_operation_table",
#         "//tensorflow/compiler/tf2xla:side_effect_util",
#         "//tensorflow/compiler/tf2xla:tf2xla_util",
#         "//tensorflow/compiler/tf2xla:xla_compiler",
#         "//tensorflow/compiler/tf2xla:xla_op_registry",
#         "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
#         "//tensorflow/compiler/tf2xla/cc:xla_ops",
#         "//xla
#         "@xla//xla:statusor",
#         "@xla//xla:union_find",
#         "@xla//xla:util",
#         "@xla//xla:xla_data_proto_cc",
#         "@xla//xla/service/graphcycles",
#         "//tensorflow/core:core_cpu",
#         "//tensorflow/core:core_cpu_internal",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:lib_internal",
#         "//tensorflow/core:protos_all_cc",
#         "//tensorflow/core/framework:bounds_check",
#         "@com_google_absl//absl/algorithm:container",
#         "@com_google_absl//absl/base",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/container:flat_hash_set",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#     ],
# )

# # openxla_xla_activity_logging_listener
# cc_library(
#     name = "openxla_xla_activity_logging_listener",
#     srcs = ["openxla_xla_activity_logging_listener.cc"],
#     deps = [
#         ":xla_activity_listener",
#         ":xla_activity_proto_cc",
#         "//tensorflow/core:lib",
#         "@com_google_absl//absl/memory",
#     ],
#     alwayslink = 1,
# )

# # openxla_jit_compilation_passes
# cc_library(
#     name = "openxla_jit_compilation_passes",
#     srcs = ["openxla_jit_compilation_pass_registration.cc"],
#     deps = [
#         ":openxla_compilation_passes",
#         ":openxla_xla_activity_logging_listener",
#         "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
#         "//tensorflow/compiler/tf2xla:mlir_bridge_pass_registration",
#         "//tensorflow/core:core_cpu_internal",
#     ] + tf_jit_compilation_passes_extra_deps(),
#     alwayslink = 1,
# )

# # openxla_xla_device
# cc_library(
#     name = "openxla_xla_device",
#     hdrs = [
#         # "xla_compile_on_demand_op.h",
#         "openxla_xla_device.h",
#         # "xla_device_context.h",
#         "xla_device_ops.h",
#     ],
#     # Public visibility is needed for external TF/XLA backends.
#     visibility = ["//visibility:public"],
#     # deps = XLA_DEVICE_DEPS + [
#     #     # ":device_compilation_profiler",
#     #     # ":openxla_jit_compilation_passes",
#     #     # ":openxla_xla_device_no_jit_rewrite_registration",
#     # ],
# )

# # openxla_xla_kernel_creator
# cc_library(
#     name = "openxla_xla_kernel_creator",
#     srcs = [
#         "openxla_xla_kernel_creator.cc",
#         "openxla_xla_kernel_creator.h",
#     ],
#     visibility = [
#         ":internal",
#         "//tensorflow/core/common_runtime/eager:__pkg__",
#     ],
#     deps = [
#         ":common",
#         ":compilability_check_util",
#         ":compilation_passes",
#         ":flags",
#         ":jit_compilation_passes",
#         "//tensorflow/compiler/jit/kernels:xla_ops_no_jit_rewrite_registration",
#         "//tensorflow/compiler/tf2xla:mlir_bridge_pass",
#         "//tensorflow/compiler/tf2xla:xla_compiler",
#         "//tensorflow/compiler/tf2xla:xla_op_registry",
#         "//tensorflow/core:core_cpu_internal",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:protos_all_cc",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:str_format",
#     ],
#     alwayslink = 1,
# )

# # openxla_xla_device_no_jit_rewrite_registration
# cc_library(
#     name = "openxla_xla_device_no_jit_rewrite_registration",
#     srcs = [
#         "xla_compile_on_demand_op.cc",
#         "xla_device.cc",
#         "xla_device_context.cc",
#         "xla_device_ops.cc",
#         "xla_ops_on_regular_devices.cc",
#         "xla_platform_info.cc",
#     ],
#     hdrs = [
#         "xla_compile_on_demand_op.h",
#         "xla_device.h",
#         "xla_device_context.h",
#         "xla_device_ops.h",
#         "xla_platform_info.h",
#     ],
#     # Public visibility is needed for external TF/XLA backends.
#     visibility = ["//visibility:public"],
#     deps = XLA_DEVICE_DEPS + [
#         ":device_compilation_cache",
#         ":device_compilation_profiler",
#         ":device_compiler_client",
#         ":device_executable_persistor",
#         ":flags_headers",
#         ":device_compiler",
#         ":xla_device_compiler_client",
#         "//tensorflow/core/tpu:tpu_defs",
#     ],
#     alwayslink = 1,
# )

# # openxla_dump_graph
# cc_library(
#     name = "openxla_dump_graph",
#     srcs = ["utils/dump_graph.cc"],
#     hdrs = ["utils/dump_graph.h"],
#     deps = [
#         ":error_util",
#         ":tensorflow",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:graph",
#         "//tensorflow/core:lib",
#         "//tensorflow/core/ir/importexport:graphdef_import",
#         "//tensorflow/core/platform:logging",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#     ],
# )

# # Linked by tensorflow core, without registration of jit compilation passes.
# cc_library(
#     name = "openxla_xla_ops_no_jit_rewrite_registration",
#     srcs = ["openxla_xla_ops.cc"],
#     hdrs = ["openxla_xla_ops.h"],
#     deps = XLA_OPS_DEPS + [
#         "//tensorflow/compiler/jit:device_compilation_cache",
#         "//tensorflow/compiler/jit:device_compilation_profiler",
#         "//tensorflow/compiler/jit:tf_graph_to_hlo_compiler",
#         "//tensorflow/compiler/jit:tf_to_hlo_compiler",
#         "//tensorflow/compiler/jit:xla_compile_util",
#         "//xla/pjrt:pjrt_client",
#         "//tensorflow/core/platform:refcount",
#     ],
#     alwayslink = 1,
# )

# # openxla_xla_ops
# cc_library(
#     name = "openxla_xla_ops",
#     hdrs = ["openxla_xla_ops.h"],
#     deps = XLA_OPS_DEPS + [
#         ":openxla_xla_ops_no_jit_rewrite_registration",
#         "//tensorflow/compiler/jit:jit_compilation_passes",
#     ],
#     alwayslink = 1,
# )

# # defs.cc/h only contains string constants, and can be included in mobile
# # builds.
# filegroup(
#     name = "openxla_mobile_srcs_no_runtime",
#     srcs = [
#         "openxla_defs.cc",
#         "openxla_defs.h",
#     ],
#     visibility = [":friends"],
# )

# openxla_device_attributes_proto
tf_proto_library(
    name = "openxla_device_attributes_proto",
    srcs = ["openxla_device_attributes.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

# openxla_types_proto
tf_proto_library(
    name = "openxla_types_proto",
    srcs = ["openxla_types.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

# openxla_full_type_proto
tf_proto_library(
    name = "openxla_full_type_proto",
    srcs = ["openxla_full_type.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [],
)

# openxla_host_compute_metadata_proto
tf_proto_library(
    name = "openxla_host_compute_metadata_proto",
    srcs = ["openxla_host_compute_metadata.proto"],
    cc_api_version = 2,
    protodeps = [
        # "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

# openxla_tensor_shape_proto
tf_proto_library(
    name = "openxla_tensor_shape_proto",
    srcs = ["openxla_tensor_shape.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

# openxla_types_proto
tf_proto_library(
    name = "openxla_types_proto",
    srcs = ["openxla_types.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

# openxla_host_compute_metadata_proto
tf_proto_library(
    name = "openxla_host_compute_metadata_proto",
    srcs = ["openxla_host_compute_metadata.proto"],
    cc_api_version = 2,
    protodeps = [
        ":openxla_tensor_shape_proto",
        ":openxla_types_proto",
    ],
    visibility = ["//visibility:public"],
)

# # openxla_sharding_util
# cc_library(
#     name = "openxla_sharding_util",
#     srcs = ["openxla_sharding_util.cc"],
#     hdrs = ["openxla_sharding_util.h"],
#     visibility = ["//visibility:public"],
#     deps = [
#         "@xla//xla:status_macros",
#         "@xla//xla/client:sharding_builder",
#         # "//tensorflow/core:core_cpu",
#         # "//tensorflow/core:core_cpu_internal",
#         # "//tensorflow/core:framework",
#         "//tensorflow/core/framework:node_def_proto_cc",
#         "//tensorflow/core:framework_internal",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:protos_all_cc",
#         "@com_google_absl//absl/strings",
#     ],
# )

# # openxla_xla_resource
# cc_library(
#     name = "openxla_xla_resource",
#     srcs = [
#         "openxla_xla_resource.cc",
#     ],
#     hdrs = [
#         "openxla_xla_resource.h",
#     ],
#     deps = [
#         ":openxla_common",
#         ":openxla_sharding_util",
#         ":xla_helpers",
#         "@xla//xla:xla_data_proto_cc",
#         "@xla//xla/client:xla_builder",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:protos_all_cc",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#     ],
#     alwayslink = 1,
# )

# # openxla_xla_argument
# cc_library(
#     name = "openxla_xla_argument",
#     srcs = [
#         "openxla_xla_argument.cc",
#     ],
#     hdrs = [
#         "openxla_xla_argument.h",
#     ],
#     deps = [
#         ":openxla_host_compute_metadata_proto",
#         ":xla_resource",
#         "@xla//xla/client:xla_builder",
#         "@xla//xla/hlo/ir:hlo",
#         "//tensorflow/core:framework",
#         "@com_google_absl//absl/types:optional",
#         "@com_google_absl//absl/types:span",
#         "@llvm-project//llvm:Support",
#     ],
#     alwayslink = 1,
# )

# openxla_layout_util
cc_library(
    name = "openxla_layout_util",
    srcs = [
        "openxla_layout_util.cc",
    ],
    hdrs = [
        "openxla_layout_util.h",
    ],
    visibility = [":friends"],
    deps = [
        ":openxla_common",
        # ":xla_argument",
        # ":xla_helpers",
        "@xla//xla:shape_util",
        "@xla//xla:statusor",
        "@xla//xla:xla_data_proto_cc",
        # "//tensorflow/core:framework",
        # "//tensorflow/core:lib",
        # "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

# # openxla_xla_compiler
# cc_library(
#     name = "openxla_xla_compiler",
#     srcs = [
#         "const_analysis.cc",
#         "graph_compiler.cc",
#         "xla_compiler.cc",
#         "xla_cpu_backend.cc",
#         "xla_op_kernel.cc",
#     ] + if_cuda_is_configured([
#         "xla_gpu_backend.cc",
#     ]) + if_rocm_is_configured([
#         "xla_gpu_backend.cc",
#     ]),
#     hdrs = [
#         "const_analysis.h",
#         "graph_compiler.h",
#         "xla_compiler.h",
#         "xla_helpers.h",
#         "xla_op_kernel.h",
#         "xla_op_registry.h",
#     ],
#     copts = tf_copts(),
#     visibility = [":friends"],
#     deps = [
#         ":common",
#         ":host_compute_metadata_proto_cc",
#         ":layout_util",
#         ":rearrange_function_argument",
#         ":sharding_util",
#         ":side_effect_util",
#         ":tf2xla_util",
#         ":xla_argument",
#         ":xla_compilation_device",
#         ":xla_context",
#         ":xla_expression",
#         ":xla_helpers",
#         ":xla_op_registry",
#         ":xla_resource",
#         "//tensorflow/compiler/jit:common",
#         "//tensorflow/compiler/jit:flags",
#         "//tensorflow/compiler/jit:shape_inference",
#         "//tensorflow/compiler/jit:xla_compile_util",
#         "//tensorflow/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
#         "//tensorflow/compiler/mlir/tf2xla/api/v0:compile_mlir_util_no_tf_dialect_passes",
#         "//tensorflow/compiler/mlir/utils:array_container_utils",
#         "@xla//xla:executable_run_options",
#         "@xla//xla
#         "@xla//xla:shape_util",
#         "@xla//xla:status_macros",
#         "@xla//xla
#         "@xla//xla:xla_data_proto_cc",
#         "@xla//xla/client:client_library",
#         "@xla//xla/client:local_client",
#         "@xla//xla/client:value_inference",
#         "@xla//xla/client:xla_builder",
#         "@xla//xla/client:xla_computation",
#         "@xla//xla/hlo/ir:hlo",
#         "@xla//xla/service:computation_placer_hdr",
#         "@xla//xla/translate/mhlo_to_hlo:layout_util",
#         "//tensorflow/core:core_cpu",
#         "//tensorflow/core:core_cpu_internal",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:lib_internal",
#         "//tensorflow/core:ops",
#         "//tensorflow/core:protos_all_cc",
#         "//tensorflow/core/tpu:tpu_defs",
#         "//tensorflow/core/util:overflow",
#         "@com_google_absl//absl/algorithm:container",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/types:optional",
#         "@com_google_absl//absl/types:span",
#         "@com_google_absl//absl/types:variant",
#     ] + if_libtpu([
#         ":xla_tpu_backend_registration",
#     ]),
#     alwayslink = 1,
# )

# # openxla_gpu_device
# cc_library(
#     name = "openxla_gpu_device",
#     srcs = ["openxla_gpu_device.cc"],
#     visibility = [":friends"],
#     deps = [
#         "@com_google_absl//absl/memory", # #include "absl/memory/memory.h"
#         "@com_google_absl//absl/strings", # #include "absl/strings/numbers.h", #include "absl/strings/str_split.h"
#         "@xla//xla/stream_executor/gpu:gpu_init", # #include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"
#         ":openxla_common", # #include "third_party/xla_client/openxla_defs.h"
#         ":openxla_flags", # #include "third_party/xla_client/openxla_flags.h"
#         ":openxla_xla_ops", # #include "third_party/xla_client/openxla_xla_ops.h"
#         ":openxla_jit_compilation_passes",
#         ":openxla_xla_device",
#         ":openxla_xla_kernel_creator",  # buildcleaner: keep
#         ":openxla_xla_device_no_jit_rewrite_registration",
#         ":openxla_layout_util",
#         ":openxla_xla_compiler",
#         "//tensorflow/compiler/tf2xla:xla_op_registry",
#         "//tensorflow/compiler/tf2xla/kernels:xla_ops",
#         "//tensorflow/core:core_cpu_internal",
#         "//tensorflow/core:lib",
#     ] + if_libtpu(
#         if_false = [
#             "@xla//xla/service:gpu_plugin",  # buildcleaner: keep
#         ],
#         if_true = [],
#     ),
#     alwayslink = 1,
# )

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
        ":conv_op_helpers",
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
        ":openxla_gpu_device",
        "@xla//xla/service:gpu_plugin_impl",
        "@xla//xla/service:gpu_plugin_stub",
        "@xla//xla/service:gpu_plugin_noncuda",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/stream_executor:cuda_platform",
        "@xla//xla/stream_executor/gpu:gpu_init",
    ]) + if_with_tpu_support([
        # "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_device",
        # "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_jit",
    ]),
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

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
    ],
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
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
        "@com_google_absl//absl/strings",
        # "@org_tensorflow//tensorflow/core:lib",
        # "@org_tensorflow//tensorflow/python/profiler/internal:profiler_pywrap_impl",
        "@tsl//tsl/platform/cloud:gcs_file_system",
        ":conv_op_helpers",
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
        "@xla//xla:literal_util",
        "@xla//xla:xla_data_cc_grpc_proto",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client:global_data",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "@xla//xla/client/lib:svd",
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