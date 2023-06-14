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
        "//torch_xla/csrc/runtime:async_task",
        "//torch_xla/csrc/runtime",
        "//torch_xla/csrc/runtime:stablehlo_helper",
        "//torch_xla/csrc/runtime:unique",
        "//torch_xla/csrc/runtime:xla_util",
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
        "//torch_xla/csrc/runtime",
        "//torch_xla/csrc/runtime:util",
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
        "//torch_xla/csrc/runtime",
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
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:tf_logging",
        "//torch_xla/csrc/runtime:util",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/core/framework:tensor_shape",
    ],
)

cc_library(
    name = "einsum_utilities",
    hdrs = ["ops/einsum_utilities.h"],
    deps = [
        "//torch_xla/csrc/runtime:debug_macros",
    ],
)

ptxla_cc_library(
    name = "aten_cpu_fallback",
    srcs = ["aten_cpu_fallback.cpp"],
    hdrs = ["aten_cpu_fallback.h"],
    deps = [
        ":function_call_tracker",
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc/runtime:metrics",
        "//torch_xla/csrc/runtime:tf_logging",
    ],
)

ptxla_cc_library(
    name = "function_call_tracker",
    srcs = ["function_call_tracker.cpp"],
    hdrs = ["function_call_tracker.h"],
    deps = [
        "//torch_xla/csrc/runtime:sys_util",
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

ptxla_cc_library(
    name = "init_python_bindings",
    srcs = ["init_python_bindings.cpp"],
    deps = [
        ":computation",
        ":device",
        ":tensor",
        ":version",
        "//torch_xla/csrc/runtime",
        "//torch_xla/csrc/runtime:mesh_service",
        "//torch_xla/csrc/runtime:metrics",
        "//torch_xla/csrc/runtime:metrics_analysis",
        "//torch_xla/csrc/runtime:metrics_reader",
        "//torch_xla/csrc/runtime:multi_wait",
        "//torch_xla/csrc/runtime:profiler",
        "//torch_xla/csrc/runtime:record_reader",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:thread_pool",
        "//torch_xla/csrc/runtime:util",
        "//torch_xla/csrc/runtime:xla_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@org_tensorflow//tensorflow/tsl/profiler/lib:traceme",
        "@org_tensorflow//tensorflow/compiler/xla/python/profiler/internal:traceme_wrapper",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_verifier",
        "@org_tensorflow//tensorflow/compiler/xla/service:sharding_propagation",
        "@org_tensorflow//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/core/platform:env",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
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
        "//torch_xla/csrc/runtime:cache",
        "//torch_xla/csrc/runtime:computation_client",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "shape_helper",
    srcs = ["shape_helper.cpp"],
    hdrs = ["shape_helper.h"],
    deps = [
        "//torch_xla/csrc/runtime:debug_macros",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "xla_data",
    srcs = ["xla_data.cpp"],
    hdrs = ["xla_data.h"],
    deps = [
        ":device",
        "//torch_xla/csrc/runtime:computation_client",
    ],
)

ptxla_cc_library(
    name = "unwrap_data",
    srcs = ["unwrap_data.cpp"],
    hdrs = ["unwrap_data.h"],
    deps = [
        ":xla_data",
        "//torch_xla/csrc/runtime:computation_client",
        "@com_google_absl//absl/types:span",
    ],
)
