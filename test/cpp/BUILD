"""Tests for PyTorch/XLA torch_xla/csrc."""

load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_library",
    "ptxla_cc_test",
)

ptxla_cc_library(
    name = "metrics_snapshot",
    srcs = ["metrics_snapshot.cpp"],
    hdrs = ["metrics_snapshot.h"],
    deps = [
        "//torch_xla/csrc/runtime:metrics",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:tf_logging",
        "//torch_xla/csrc/runtime:util",
    ],
)

ptxla_cc_library(
    name = "torch_xla_test",
    srcs = ["torch_xla_test.cpp"],
    hdrs = ["torch_xla_test.h"],
    deps = [
        ":metrics_snapshot",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc/runtime:tf_logging",
        "//torch_xla/csrc:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "cpp_test_util",
    srcs = ["cpp_test_util.cpp"],
    hdrs = ["cpp_test_util.h"],
    deps = [
        "//torch_xla/csrc/runtime:runtime",
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc:tensor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_test(
    name = "test_ir",
    srcs = ["test_ir.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "@com_google_googletest//:gtest_main",
    ],
)

ptxla_cc_test(
    name = "test_lazy",
    srcs = ["test_lazy.cpp"],
    deps = [
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
    ],
)

ptxla_cc_test(
    name = "test_replication",
    srcs = ["test_replication.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc/runtime:runtime",
        "//torch_xla/csrc/runtime:debug_macros",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "//torch_xla/csrc:thread_pool",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
        "@xla//xla/client:xla_builder",
    ],
)

ptxla_cc_test(
    name = "test_tensor",
    srcs = [
        "test_symint.cpp",
        "test_tensor.cpp",
    ],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "@com_google_googletest//:gtest_main",
    ],
)

# Disable this test since it is flaky on upstream
# ptxla_cc_test(
#     name = "test_xla_backend_intf",
#     srcs = ["test_xla_backend_intf.cpp"],
#     deps = [
#         ":cpp_test_util",
#         "//torch_xla/csrc:tensor",
#         "@com_google_googletest//:gtest_main",
#     ],
# )

ptxla_cc_test(
    name = "test_xla_sharding",
    srcs = ["test_xla_sharding.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc/runtime:env_vars",
        "//torch_xla/csrc/runtime:sys_util",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/tsl/profiler/utils:session_manager",
    ],
)

# This tets is very large so it's split into shards.
# To make it run fast, please add new shards when needed.
[
    ptxla_cc_test(
        name = test[:-len(".cpp")],
        size = "enormous",
        srcs = [test],
        deps = [
            ":cpp_test_util",
            ":torch_xla_test",
            "//torch_xla/csrc/runtime:metrics",
            "//torch_xla/csrc:tensor",
            "//torch_xla/csrc:aten_cuda_functions",
            "@com_google_googletest//:gtest_main",
            "@xla//xla:permutation_util",
        ],
    )
    for test in glob(["test_aten_xla_tensor*cpp"])
]
