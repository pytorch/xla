load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_library",
)

genrule(
    name = "gen_lazy_tensor",
    srcs = ["//codegen:xla_native_functions.yaml"],
    outs = [
        "LazyIr.h",
        "LazyNonNativeIr.h",
        "RegisterAutogradXLA.cpp",
        "RegisterXLA.cpp",
        "XLANativeFunctions.cpp",
        "XLANativeFunctions.h",
    ],
    cmd = ";".join([
        "$(location //codegen:lazy_tensor_generator) $(location //codegen:lazy_tensor_generator).runfiles $< $(RULEDIR)",
        "$(location //codegen:fix_includes) $(OUTS)",
    ]),
    tags = [
        "local",
        "no-remote-exec",
    ],
    tools = [
        "//codegen:fix_includes",
        "//codegen:lazy_tensor_generator",
    ],
)

ptxla_cc_library(
    name = "tensor",
    srcs = [
        "aten_autograd_ops.cpp",
        "aten_xla_bridge.cpp",
        "aten_xla_type.cpp",
        "batch_norm.cpp",
        "convert_ops.cpp",
        "convolution.cpp",
        "cross_replica_reduces.cpp",
        "data_ops.cpp",
        "debug_util.cpp",
        "elementwise.cpp",
        "helpers.cpp",
        "ir_dump_util.cpp",
        "matrix.cpp",
        "nll_loss.cpp",
        "nms_op.cpp",
        "op_by_op_executor.cpp",
        "pooling.cpp",
        "random.cpp",
        "reduction.cpp",
        "resize_ops.cpp",
        "softmax_builder.cpp",
        "tensor.cpp",
        "tensor_impl.cpp",
        "tensor_methods.cpp",
        "tensor_ops.cpp",
        "tensor_util.cpp",
        "token_handler.cpp",
        "torch_util.cpp",
        "view.cpp",
        "xla_backend_impl.cpp",
        "xla_graph_executor.cpp",
        "xla_lower_util.cpp",
        "xla_op_builder.cpp",
        "xla_sharding_util.cpp",
        ":RegisterAutogradXLA.cpp",
        ":RegisterXLA.cpp",
        ":XLANativeFunctions.cpp",
    ] + glob(["ops/*.cpp"]),
    hdrs = [
        "aten_autograd_ops.h",
        "aten_xla_bridge.h",
        "batch_norm.h",
        "convert_ops.h",
        "convolution.h",
        "cross_replica_reduces.h",
        "data_ops.h",
        "debug_util.h",
        "elementwise.h",
        "generated_file_include.h",
        "helpers.h",
        "ir_dump_util.h",
        "matrix.h",
        "nll_loss.h",
        "nms_op.h",
        "op_by_op_executor.h",
        "pooling.h",
        "random.h",
        "reduction.h",
        "resize_ops.h",
        "softmax_builder.h",
        "tensor.h",
        "tensor_impl.h",
        "tensor_methods.h",
        "tensor_ops.h",
        "tensor_util.h",
        "token_handler.h",
        "torch_util.h",
        "view.h",
        "xla_backend_impl.h",
        "xla_graph_executor.h",
        "xla_lower_util.h",
        "xla_op_builder.h",
        "xla_sharding_util.h",
        ":LazyIr.h",
        ":LazyNonNativeIr.h",
        ":XLANativeFunctions.h",
    ] + glob(["ops/*.h"]),
    deps = [
        ":aten_cpu_fallback",
        ":computation",
        ":device",
        ":einsum_utilities",
        ":ir",
        ":ir_builder",
        ":layout_manager",
        ":shape_builder",
        ":shape_helper",
        "//third_party/xla_client:async_task",
        "//third_party/xla_client:runtime",
        "//third_party/xla_client:unique",
        "//third_party/xla_client:xla_util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/tf2xla/kernels:conv_op_helpers",
        "@org_tensorflow//tensorflow/compiler/xla:comparison_util",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:permutation_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:arithmetic",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:comparators",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:constants",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:logdet",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:math",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:pooling",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:slicing",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:sorting",
        "@org_tensorflow//tensorflow/compiler/xla/client/lib:svd",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor:dnn",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/framework:tensor_shape",
        "@org_tensorflow//tensorflow/core/lib/core:errors",
        "@org_tensorflow//tensorflow/core/lib/gtl:inlined_vector",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
        "@org_tensorflow//third_party/python_runtime:headers",
        "@pybind11//:pybind11_embed",  # libpyton
    ],
)

ptxla_cc_library(
    name = "device",
    srcs = ["device.cpp"],
    hdrs = ["device.h"],
    deps = [
        "//third_party/xla_client:runtime",
        "//third_party/xla_client:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "computation",
    srcs = ["computation.cpp"],
    hdrs = ["computation.h"],
    deps = [
        "//third_party/xla_client:runtime",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    ],
)

ptxla_cc_library(
    name = "layout_manager",
    srcs = ["layout_manager.cpp"],
    hdrs = ["layout_manager.h"],
    deps = [
        ":device",
        "//third_party/xla_client:debug_macros",
        "//third_party/xla_client:sys_util",
        "//third_party/xla_client:tf_logging",
        "//third_party/xla_client:util",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/core/framework:tensor_shape",
    ],
)

cc_library(
    name = "einsum_utilities",
    hdrs = ["ops/einsum_utilities.h"],
    deps = [
        "//third_party/xla_client:debug_macros",
    ],
)

ptxla_cc_library(
    name = "aten_cpu_fallback",
    srcs = ["aten_cpu_fallback.cpp"],
    hdrs = ["aten_cpu_fallback.h"],
    deps = [
        ":function_call_tracker",
        "//third_party/xla_client:debug_macros",
        "//third_party/xla_client:metrics",
        "//third_party/xla_client:tf_logging",
    ],
)

ptxla_cc_library(
    name = "function_call_tracker",
    srcs = ["function_call_tracker.cpp"],
    hdrs = ["function_call_tracker.h"],
    deps = [
        "//third_party/xla_client:sys_util",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core/platform:stacktrace",
    ],
)

ptxla_cc_library(
    name = "ir_builder",
    hdrs = ["ir_builder.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "shape_builder",
    srcs = ["shape_builder.cpp"],
    hdrs = ["shape_builder.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:types",
    ],
)

cc_library(
    name = "version",
    srcs = ["version.cpp"],
    hdrs = ["version.h"],
)

load("//tensorflow/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

# enable_gpu
bool_flag(
    name = "enable_gpu",
    build_setting_default = True,
)

# gpu_enabled
config_setting(
    name = "gpu_enabled",
    flag_values = {
        ":enable_gpu": "True",
    },
)

# enable_tpu
bool_flag(
    name = "enable_tpu",
    build_setting_default = True,
)

# tpu_enabled
config_setting(
    name = "tpu_enabled",
    flag_values = {
        ":enable_tpu": "True",
    },
)

# enable_plugin_device
bool_flag(
    name = "enable_plugin_device",
    build_setting_default = False,
)

# plugin_device_enabled
config_setting(
    name = "plugin_device_enabled",
    flag_values = {
        ":enable_plugin_device": "True",
    },
)

# ptxla_cuda_pip_rpaths
# If this flag is enabled, it sets RPATH on the xla_extension to values that are suitable for
# finding NVIDIA's CUDA libraries when they are installed as pip packages.
bool_flag(
    name = "ptxla_cuda_pip_rpaths",
    build_setting_default = False,
)

# use_ptxla_cuda_pip_rpaths
config_setting(
    name = "use_ptxla_cuda_pip_rpaths",
    flag_values = {
        ":ptxla_cuda_pip_rpaths": "True",
    },
)

# link_gpu_plugin
# TODO(phawkins): the configuration settings here are overly confusing. The right fix is to split
# xla_extension.so so that each backend is a separate plugin, however that must wait for a clean
# ABI separation between devices.
config_setting(
    name = "link_gpu_plugin",
    define_values = {"xla_python_enable_gpu": "true"},
)

# xla_extension
tsl_pybind_extension(
    name = "xla_extension",
    srcs = [
        "xla.cc",
    ],
    defines = select({
        ":gpu_enabled": ["XLA_PYTHON_ENABLE_GPU=1"],
        "//conditions:default": [],
    }) + select({
        ":tpu_enabled": ["XLA_PYTHON_ENABLE_TPU=1"],
        "//conditions:default": [],
    }) + select({
        ":plugin_device_enabled": ["XLA_PYTHON_ENABLE_PLUGIN_DEVICE=1"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":use_ptxla_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../nvidia/cuda_cupti/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cublas/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cufft/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cudnn/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cusolver/lib",
        ],
        "//conditions:default": [],
    }),
    pytype_deps = [
        "//third_party/py/numpy", # not sure the current numpy path
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/python:custom_call_sharding",
        "@org_tensorflow//tensorflow/compiler/xla/python:dlpack",
        "@org_tensorflow//tensorflow/compiler/xla/python:jax_jit",
        "@org_tensorflow//tensorflow/compiler/xla/python:mlir",
        "@org_tensorflow//tensorflow/compiler/xla/python:ops",
        "@org_tensorflow//tensorflow/compiler/xla/python:outfeed_receiver_py",
        "@org_tensorflow//tensorflow/compiler/xla/python:pjit",
        "@org_tensorflow//tensorflow/compiler/xla/python:pmap_lib",
        "@org_tensorflow//tensorflow/compiler/xla/python:pprof_profile_builder",
        "@org_tensorflow//tensorflow/compiler/xla/python:profiler",
        "@org_tensorflow//tensorflow/compiler/xla/python:py_client",
        "@org_tensorflow//tensorflow/compiler/xla/python:python_ref_manager",
        "@org_tensorflow//tensorflow/compiler/xla/python:pytree",
        "@org_tensorflow//tensorflow/compiler/xla/python:status_casters",
        "@org_tensorflow//tensorflow/compiler/xla/python:traceback",
        "@org_tensorflow//tensorflow/compiler/xla/python:transfer_guard_lib",
        "@org_tensorflow//tensorflow/compiler/xla/python:types",
        "@org_tensorflow//tensorflow/compiler/xla/python:util",
        "@org_tensorflow//tensorflow/compiler/xla/python:weakref_lru_cache",
        "@org_tensorflow//tensorflow/compiler/xla/python:xla_compiler",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:interpreter_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_api",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_compiler",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed:client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/distributed:service",
        "@org_tensorflow//tensorflow/compiler/xla/python/ifrt",
        "@org_tensorflow//tensorflow/compiler/xla/python/pjrt_ifrt",
        "@org_tensorflow//tensorflow/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "@org_tensorflow//tensorflow/tsl/platform:logging",
        "@org_tensorflow//tensorflow/tsl/python/lib/core:numpy",
        "@org_tensorflow//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + select({
        ":gpu_enabled": [
            "@org_tensorflow//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        ],
        "//conditions:default": [],
    }) + select({
        ":link_gpu_plugin": [
            "@org_tensorflow//tensorflow/compiler/xla/python:gpu_plugin_deps",
        ],
        "//conditions:default": [],
    }) + select({
        ":tpu_enabled": [
            "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
            "@org_tensorflow//tensorflow/compiler/xla/pjrt:tpu_client",
        ],
        "//conditions:default": [],
    }) + select({
        ":plugin_device_enabled": [
            "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_plugin_device_client",
        ],
        "//conditions:default": [],
    }),
)

# xla_client
pytype_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":xla_extension"],
)

ptxla_cc_library(
    name = "init_python_bindings",
    srcs = ["init_python_bindings.cpp"],
    deps = [
        ":xla_client",
        ":computation",
        ":device",
        ":tensor",
        ":version",
        "//third_party/xla_client:runtime",
        "//third_party/xla_client:mesh_service",
        "//third_party/xla_client:metrics",
        "//third_party/xla_client:metrics_analysis",
        "//third_party/xla_client:metrics_reader",
        "//third_party/xla_client:multi_wait",
        "//third_party/xla_client:profiler",
        "//third_party/xla_client:record_reader",
        "//third_party/xla_client:sys_util",
        "//third_party/xla_client:thread_pool",
        "//third_party/xla_client:util",
        "//third_party/xla_client:xla_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@org_tensorflow//tensorflow/compiler/xla/python/profiler/internal:traceme_wrapper",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_verifier",
        "@org_tensorflow//tensorflow/compiler/xla/service:sharding_propagation",
        "@org_tensorflow//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:env",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
        "@org_tensorflow//tensorflow/python/profiler/internal:profiler_pywrap_impl",
    ],
)

ptxla_cc_library(
    name = "ir",
    srcs = [
        "ir.cpp",
        "lowering_context.cpp",
    ],
    hdrs = [
        "ir.h",
        "lowering_context.h",
    ],
    deps = [
        ":computation",
        ":device",
        ":shape_helper",
        ":unwrap_data",
        ":xla_data",
        "//third_party/xla_client:cache",
        "//third_party/xla_client:computation_client",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "shape_helper",
    srcs = ["shape_helper.cpp"],
    hdrs = ["shape_helper.h"],
    deps = [
        "//third_party/xla_client:debug_macros",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "xla_data",
    srcs = ["xla_data.cpp"],
    hdrs = ["xla_data.h"],
    deps = [
        ":device",
        "//third_party/xla_client:computation_client",
    ],
)

ptxla_cc_library(
    name = "unwrap_data",
    srcs = ["unwrap_data.cpp"],
    hdrs = ["unwrap_data.h"],
    deps = [
        ":xla_data",
        "//third_party/xla_client:computation_client",
        "@com_google_absl//absl/types:span",
    ],
)
