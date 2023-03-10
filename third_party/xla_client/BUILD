load("@tsl//tsl:tsl.bzl", "if_with_tpu_support")
load(":PyTorchXLA.bzl", "ptxla_cc_shared_object")

load(
    "@tsl//tsl/platform/default:build_config.bzl",
    "tf_proto_library_cc",
)
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files([
    "openxla_version_script.lds", # this file has been added into third_party/xla_client
    "openxla_exported_symbols.lds", # this file has been added into third_party/xla_client
])

tf_proto_library_cc(
    name = "mesh_service_proto",
    srcs = ["mesh_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//third_party/protobuf/tpu:topology_proto",
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
    alwayslink = True,
)

# used in libxla_computation_client.so
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
        # ":mesh_service",
        ":metrics_analysis",
        ":metrics_reader",
        ":metrics",
        ":multi_wait",
        # ":profiler",
        ":record_reader",
        ":sys_util",
        ":tf_logging",
        ":thread_pool",
        ":triggered_task",
        ":types",
        ":unique",
        ":util",
        ":xla_util",
        ":grpc++",
        ":client_session",
        ":scope",
        ":xla_cpu_device",
        "@xla//xla:debug_options_flags",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_proto_cc",
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
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service/spmd:spmd_partitioner",
        ":core_cpu",
        ":framework_internal",
        ":lib",
        # "@org_tensorflow//tensorflow/core:protos_all_cc", # didn't find defination # "//tensorflow/tsl/protobuf:protos_all_cc_impl",
        ":server_lib",
        ":grpc_runtime",
        ":data_flow",
        # "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto_cc", # didn't find this one
        "@xla//xla/stream_executor:stream_executor_impl",
        "@tsl//tsl/profiler/rpc:profiler_server_impl",
        "@tsl//tsl/profiler/rpc/client:profiler_client",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        ":xla_gpu_device",
        "@xla//xla/stream_executor:cuda_platform",
    ]) + if_with_tpu_support([
        ":xla_tpu_device",
        ":xla_tpu_jit",
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
        ":lib",
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

ptxla_cc_shared_object(
    name = "libxla_computation_client.so",
    linkopts = select({
        # "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location :openxla_version_script.lds)",
        ],
    }),
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
        "@xla//xla/rpc:grpc_stub",
        ":lib",
        "@tsl//tsl/platform/cloud:gcs_file_system",
        # "@org_tensorflow//tensorflow/python/profiler/internal:profiler_pywrap_impl", # use tensorflow-profiler-plugin
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





########################################################################
# jit ----- 
# -- solution 0: ask some JIT export in TF
# -- solution 1: mlir/tfrt also have jit, so we might could get jit from libtpu and from pjrt then from tfrt, get jit
# -- solution 2: copy jit folder directly 
# -- solution 3: ask some JIT expert to do this, or some export from JAX.JIT
# Target that bundles up the XLA CPU and GPU JIT devices.

# load("@xla//xla/stream_executor:build_defs.bzl", "if_cuda_or_rocm")

# cc_library(
#     name = "jit",
#     visibility = [
#         -- ":friends",
#         -- "//learning/tfx:__subpackages__",
#     ],
#     deps = [
#         -- ":xla_cpu_device",
#         -- ":xla_cpu_jit",
#         -- "//tensorflow/compiler/plugin",
#     ] + if_cuda_or_rocm([
#         -- ":xla_gpu_device",
#         -- ":xla_gpu_jit",
#     ]) + if_with_tpu_support([
#         -- ":xla_tpu_device",
#         -- ":xla_tpu_jit",
#     ]),
#     alwayslink = 1,
# )

# cc_library(
#     name = "xla_cpu_jit",
#     visibility = ["//visibility:public"],
#     deps = [
#         "//third_party/jit:jit_compilation_passes",
#         "//third_party/jit:xla_kernel_creator",  # buildcleaner: keep
#         "//third_party/jit/kernels:xla_ops",
#         "//third_party/tf2xla/kernels:xla_dummy_ops",
#         "//third_party/tf2xla/kernels:xla_ops",
#     ] + if_libtpu(
#         if_false = ["@xla//xla/service:cpu_plugin"],
#         if_true = [],
#     ),
#     alwayslink = 1,
# )

# cc_library(
#     name = "xla_gpu_jit",
#     visibility = ["//visibility:public"],
#     deps = if_cuda_or_rocm([
#         "//third_party/jit:jit_compilation_passes",
#         "//third_party/jit:xla_kernel_creator",  # buildcleaner: keep
#         "//third_party/jit/kernels:xla_ops",
#         "//third_party/tf2xla/kernels:xla_ops",
#         "//third_party/tf2xla/kernels:xla_dummy_ops",
#         "@xla//xla/service:gpu_plugin",
#     ]),
#     alwayslink = 1,
# )

cc_library(
    name = "xla_tpu_jit",
    visibility = ["//visibility:public"],
    deps = if_libtpu([
        ":tpu_rewrite_pass_registration",
        ":configure_tpu_embedding_rewrite_registration",
        "@xla//xla/stream_executor/tpu:tpu_transfer_manager",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        "//third_party/jit:common",
        "//third_party/jit:flags",
        "//third_party/jit:jit_compilation_passes",
        "//third_party/jit:xla_device",
        "//third_party/jit:xla_kernel_creator",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
        "//third_party/jit/kernels:xla_ops",
        "//third_party/tf2xla:layout_util",
        "//third_party/tf2xla:xla_compiler",
        "//third_party/tf2xla:xla_op_registry",
        "//third_party/tf2xla/kernels:xla_ops",
        ":core_cpu_internal",
        ":lib",
    ] + if_libtpu(
        if_false = [
            "@xla//xla/service:cpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_device",
    srcs = ["xla_gpu_device.cc"],
    visibility = [":friends"],
    deps = [
        "//third_party/jit:common",
        "//third_party/jit:flags",
        "//third_party/jit:jit_compilation_passes",
        "//third_party/jit:xla_device",
        "//third_party/jit:xla_kernel_creator",  # buildcleaner: keep
        "//third_party/jit:xla_device_no_jit_rewrite_registration",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//third_party/jit/kernels:xla_ops",
        "//third_party/tf2xla:layout_util",
        "//third_party/tf2xla:xla_compiler",
        "//third_party/tf2xla:xla_op_registry",
        "//third_party/tf2xla/kernels:xla_ops",
        "@xla//xla/stream_executor/gpu:gpu_init",
        ":core_cpu_internal",
        ":lib",
    ] + if_libtpu(
        if_false = [
            "@xla//xla/service:gpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_device",
    srcs = ["xla_tpu_device.cc"],
    hdrs = ["xla_tpu_device.h"],
    visibility = [":friends"],
    deps = [
        "//third_party/jit:xla_device",
        "//third_party/jit:xla_kernel_creator",  # buildcleaner: keep
        "@com_google_absl//absl/types:optional",
        "//third_party/jit/kernels:xla_ops",
        "//third_party/tf2xla:common",
        "//third_party/tf2xla:layout_util",
        "//third_party/tf2xla:tf2xla_util",
        "//third_party/tf2xla:xla_helpers",
        "//third_party/tf2xla:xla_op_registry",
        ":framework_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/platform:status",
        "@xla//xla/stream_executor/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_node_device_util",
        "//tensorflow/core/tpu:virtual_device",
        "@xla//xla/stream_executor/tpu:c_api_conversions",
        "@xla//xla/stream_executor/tpu:status_helper",
        "@xla//xla/stream_executor/tpu:tpu_executor_base",
        "@xla//xla/stream_executor/tpu:tpu_node_context",
        "@xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@xla//xla/stream_executor/tpu:tpu_stream_interface",
    ] + if_static([
        "//tensorflow/core/common_runtime:copy_tensor",
        ":jit_compilation_passes",
    ]),
    alwayslink = 1,
)

# cc_library(
#     name = "plugin",
#     deps = [
#         #"//tensorflow/compiler/plugin/example:example_lib",
#     ],
# )

tf_cuda_library(
    name = "framework_internal",
    srcs = [":framework_internal_private_headers"],
    hdrs = [":framework_internal_public_headers"],
    deps = [
        ":framework_internal_headers_lib",
        "//third_party/eigen3",
        ":lib",
    ] + if_static(
        extra_deps = [
            ":framework_internal_impl",
            "@com_google_protobuf//:protobuf",
        ],
        otherwise = [
            "//tensorflow/core/util:managed_stack_trace",
            "@com_google_protobuf//:protobuf_headers",
        ],
    ),
    alwayslink = 1,
)

cc_header_only_library(
    name = "framework_internal_headers_lib",
    # Fully depend on external repositories, because identifying the headers
    # is fragile.
    extra_deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    deps = [
        ":lib",
        ":lib_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

# //tensorflow/core/lib
# This build rule (along with :lib_internal, :framework, and
# :framework_internal) purposefully omits the definitions of many declared
# symbols, which are included in //tensorflow:libtensorflow_framework.so. Using
# tf_cc_test and tf_cc_binary will include the necessary symbols.
cc_library(
    name = "lib",
    hdrs = [
        # TODO(rmlarsen): Remove bfloat16.h once dependency in third_party/swift is updated.
        "//tensorflow/core/lib/bfloat16:bfloat16.h",
        "//tensorflow/core/lib/core:legacy_lib_core_headers",
        "//tensorflow/tsl/lib/core:legacy_lib_core_headers",
        "//tensorflow/core/lib/gtl:legacy_lib_gtl_headers",
        "//tensorflow/core/lib/hash:legacy_lib_hash_all_headers",
        "//tensorflow/core/lib/histogram:legacy_lib_histogram_all_headers",
        "//tensorflow/core/lib/io:legacy_lib_io_headers",
        "//tensorflow/core/lib/math:math_util.h",
        "//tensorflow/tsl/lib/math:math_util.h",
        "//tensorflow/core/lib/monitoring:legacy_lib_monitoring_lib_headers",
        "//tensorflow/core/lib/random:legacy_lib_random_headers",
        "//tensorflow/core/lib/strings:legacy_lib_string_headers",
        "//tensorflow/core/platform:lib_hdrs",
        "//tensorflow/core/util:lib_hdrs",
        "//tensorflow/tsl/lib/io:legacy_lib_io_headers",
        "//tensorflow/tsl/platform:lib_hdrs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":lib_internal",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "grpc++",
    visibility = ["//visibility:public"],
    deps = select({
        ":linux_s390x": ["@com_github_grpc_grpc//:grpc++_unsecure"],
        "//conditions:default": ["@com_github_grpc_grpc//:grpc++"],
    }),
)

cc_library_with_android_deps(
    name = "client_session",
    srcs = ["client/client_session.cc"],
    hdrs = ["client/client_session.h"],
    android_deps = ["//tensorflow/core:portable_tensorflow_lib"],
    common_deps = [
        ":ops",
        ":scope",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        ":lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library_with_android_deps(
    name = "scope",
    srcs = [
        "framework/scope.cc",
        "framework/scope_internal.h",
    ],
    hdrs = ["framework/scope.h"],
    android_deps = ["//tensorflow/core:portable_tensorflow_lib"],
    common_deps = [
        ":ops",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        ":lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

alias(
    name = "core_cpu",
    actual = "//tensorflow/core/common_runtime:core_cpu",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework_internal",
        ":lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# @org_tensorflow//tensorflow/core/distributed_runtime/rpc
cc_library(
    name = "grpc_runtime",
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":grpc_server_lib",
        ":grpc_session",
    ],
)

# @org_tensorflow//tensorflow/core/kernels
cc_library(
    name = "data_flow",
    deps = [
        ":barrier_ops",
        ":conditional_accumulator_base_op",
        ":conditional_accumulator_op",
        ":dynamic_partition_op",
        ":dynamic_stitch_op",
        ":fifo_queue_op",
        ":lookup_table_init_op",
        ":lookup_table_op",
        ":map_stage_op",
        ":padding_fifo_queue_op",
        ":priority_queue_op",
        ":queue_ops",
        ":random_shuffle_queue_op",
        ":record_input_op",
        ":session_ops",
        ":sparse_conditional_accumulator_op",
        ":stack_ops",
        ":stage_op",
        ":tensor_array_ops",
    ],
)

# //tensorflow/core/tpu/graph_rewrite
cc_library(
    name = "tpu_rewrite_pass_registration",
    srcs = ["tpu_rewrite_pass_registration.cc"],
    deps = [
        ":combine_tpu_embedding_load_retrieve_pass",
        ":distributed_tpu_configuration_rewrite_pass",
        ":distributed_tpu_rewrite_pass",
        ":encapsulate_tpu_computations_pass",
        ":tpu_embedding_software_deduplication_rewrite_pass",
        ":update_tpu_embedding_ops_passes",
        ":variable_merger_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

# //tensorflow/core/tpu/graph_rewrite
cc_library(
    name = "configure_tpu_embedding_rewrite_registration",
    srcs = ["configure_tpu_embedding_rewrite_registration.cc"],
    deps = [
        ":configure_tpu_embedding_rewrite_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

alias(
    name = "core_cpu_internal",
    actual =
        "//tensorflow/core/common_runtime:core_cpu_internal",
)