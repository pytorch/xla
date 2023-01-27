load("@org_tensorflow//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_with_tpu_support",
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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["@org_tensorflow//tensorflow:internal"])

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
        "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto",
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
        ":computation_client_impl",
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
        "@org_tensorflow//tensorflow/core/platform/cloud:gcs_file_system",
        "@org_tensorflow//tensorflow/python/profiler/internal:profiler_pywrap_impl",
    ],
)

cc_library(
    name = "computation_client_impl",
    srcs = [
        "computation_client.cc",
        "env_vars.cc",
        "mesh_service.cc",
        "metrics.cc",
        "metrics_analysis.cc",
        "metrics_reader.cc",
        "multi_wait.cc",
        "nccl_distributed.cc",
        "pjrt_computation_client.cc",
        "profiler.cc",
        "record_reader.cc",
        "sys_util.cc",
        "tf_logging.cc",
        "thread_pool.cc",
        "triggered_task.cc",
        "util.cc",
        "xla_util.cc",
        "pjrt_computation_client.cc",
        "xrt_computation_client.cc",
        "xrt_local_service.cc",
        "xrt_session.cc",
        "xrt_session_cache.cc",
    ],
    hdrs = [
        "cache.h",
        "computation_client.h",
<<<<<<< HEAD
        "debug_macros.h",
        "env_vars.h",
        "mesh_service.h",
        "metrics.h",
        "metrics_analysis.h",
        "metrics_reader.h",
        "multi_wait.h",
        "nccl_distributed.h",
        "pjrt_computation_client.h",
        "profiler.h",
        "record_reader.h",
        "sys_util.h",
        "tf_logging.h",
        "thread_pool.h",
        "triggered_task.h",
        "types.h",
        "unique.h",
        "util.h",
        "xla_util.h",
        "pjrt_computation_client.h",
        "xrt_computation_client.h",
        "xrt_local_service.h",
        "xrt_session.h",
        "xrt_session_cache.h",
    ],
    deps = [
        ":cache",
        ":debug_macros",
        ":env_vars",
        ":mesh_service",
        ":metrics",
        ":sys_util",
        ":tf_logging",
        ":triggered_task",
        ":types",
        ":unique",
        ":xla_util",
        ":xrt_local_service",
        ":xrt_session_cache",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
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
        "@org_tensorflow//tensorflow/core/profiler/rpc:profiler_server_impl",
        "@org_tensorflow//tensorflow/core/profiler/rpc/client:profiler_client",
        "@org_tensorflow//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor:cuda_platform",
    ]) + if_with_tpu_support([
        "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_device",
        "@org_tensorflow//tensorflow/compiler/jit:xla_tpu_jit",
    ]),
    alwayslink = 1,
)

xla_cc_test(
    name = "pjrt_computation_client_test",
    srcs = ["pjrt_computation_client_test.cc"],
    deps = [
        ":computation_client_impl",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/core/platform:logging",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
)

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
        ":tf_version_script.lds",
    ],
)