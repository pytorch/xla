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
        "aten_fallback.cpp",
        "aten_xla_bridge.cpp",
        "aten_xla_type.cpp",
        "autocast_mode.cpp",
        "batch_norm.cpp",
        "convert_ops.cpp",
        "convolution.cpp",
        "convolution_helper.cpp",
        "cross_replica_reduces.cpp",
        "data_ops.cpp",
        "debug_util.cpp",
        "dl_convertor.cpp",
        "elementwise.cpp",
        "helpers.cpp",
        "ir_dump_util.cpp",
        "matrix.cpp",
        "nll_loss.cpp",
        "pooling.cpp",
        "quant_util.cpp",
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
        "xla_manual_registration.cpp",
        ":RegisterAutogradXLA.cpp",
        ":RegisterXLA.cpp",
        ":XLANativeFunctions.cpp",
    ] + glob(["ops/*.cpp"]),
    hdrs = [
        "aten_autograd_ops.h",
        "aten_fallback.h",
        "aten_cuda_functions.h",
        "aten_xla_bridge.h",
        "batch_norm.h",
        "convert_ops.h",
        "convolution.h",
        "convolution_helper.h",
        "cross_replica_reduces.h",
        "data_ops.h",
        "debug_util.h",
        "dl_convertor.h",
        "elementwise.h",
        "generated_file_include.h",
        "helpers.h",
        "ir_dump_util.h",
        "matrix.h",
        "nll_loss.h",
        "pooling.h",
        "quant_util.h",
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
        ":device",
        ":dtype",
        ":einsum_utilities",
        ":function_call_tracker",
        ":ir",
        ":ir_builder",
        ":layout_manager",
        ":shape_builder",
        ":shape_helper",
        ":version",
        "//torch_xla/csrc/runtime",
        "//torch_xla/csrc/runtime:stablehlo_helper",
        "//torch_xla/csrc/runtime:xla_util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:inlined_vector",
        "@xla//xla:comparison_util",
        "@xla//xla:literal_util",
        "@xla//xla:permutation_util",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client/lib:arithmetic",
        "@xla//xla/client/lib:comparators",
        "@xla//xla/client/lib:constants",
        "@xla//xla/client/lib:logdet",
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:pooling",
        "@xla//xla/client/lib:self_adjoint_eig",
        "@xla//xla/client/lib:slicing",
        "@xla//xla/client/lib:sorting",
        "@xla//xla/client/lib:svd",
        "@xla//xla/hlo/pass:hlo_pass_pipeline",
        "@xla//xla/stream_executor:dnn",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@local_config_python//:python_headers",
        "@pybind11//:pybind11_embed",  # libpyton
    ],
)

ptxla_cc_library(
    name = "device",
    srcs = ["device.cpp"],
    hdrs = ["device.h"],
    deps = [
        "//torch_xla/csrc/runtime:tf_logging",
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@xla//xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "dtype",
    srcs = ["dtype.cpp"],
    hdrs = ["dtype.h"],
    deps = [
        "//torch_xla/csrc:device",
        "//torch_xla/csrc/runtime:tf_logging",
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
        "@xla//xla:shape_util",
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
    name = "function_call_tracker",
    srcs = ["function_call_tracker.cpp"],
    hdrs = ["function_call_tracker.h"],
    deps = [
        "//torch_xla/csrc/runtime:sys_util",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:stacktrace",
    ],
)

ptxla_cc_library(
    name = "ir_builder",
    hdrs = ["ir_builder.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@xla//xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "shape_builder",
    srcs = ["shape_builder.cpp"],
    hdrs = ["shape_builder.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@xla//xla:shape_util",
        "@xla//xla:types",
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
        ":device",
        ":dtype",
        ":tensor",
        ":version",
        "//torch_xla/csrc/runtime",
        "//torch_xla/csrc/runtime:pjrt_computation_client",
        "//torch_xla/csrc/runtime:metrics",
        "//torch_xla/csrc/runtime:metrics_analysis",
        "//torch_xla/csrc/runtime:metrics_reader",
        "//torch_xla/csrc/runtime:profiler",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:util",
        "//torch_xla/csrc/runtime:xla_coordinator",
        "//torch_xla/csrc/runtime:xla_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:variant",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@xla//xla/python/profiler/internal:traceme_wrapper",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_pass_pipeline",
        "@xla//xla/service:hlo_verifier",
        "@xla//xla/service:sharding_propagation",
        "@xla//xla/service/spmd:spmd_partitioner",
    ],
)

ptxla_cc_library(
    name = "ir",
    srcs = [
        "dynamic_shape_detector.cpp",
        "ir.cpp",
        "lowering_context.cpp",
        "stack_frame_index_builder.cpp",
    ],
    hdrs = [
        "dynamic_shape_detector.h",
        "ir.h",
        "lowering_context.h",
        "stack_frame_index_builder.h",
    ],
    deps = [
        ":device",
        ":shape_helper",
        ":unwrap_data",
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
        "@xla//xla/client:xla_builder",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "//torch_xla/csrc/runtime:sys_util",
        "@tsl//tsl/platform:env"
    ],
)

ptxla_cc_library(
    name = "unwrap_data",
    srcs = ["unwrap_data.cpp"],
    hdrs = ["unwrap_data.h"],
    deps = [
        "//torch_xla/csrc/runtime:computation_client",
        "@com_google_absl//absl/types:span",
    ],
)

ptxla_cc_library(
    name = "aten_cuda_functions",
    srcs = ["aten_cuda_functions.cpp"],
    hdrs = ["aten_cuda_functions.h"],
    deps = [
        "@local_config_python//:python_headers",
        "@pybind11//:pybind11_embed",
    ],
)
