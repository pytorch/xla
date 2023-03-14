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
        "//third_party/xla_client:metrics",
        "//third_party/xla_client:sys_util",
        "//third_party/xla_client:tf_logging",
        "//third_party/xla_client:util",
    ],
)

ptxla_cc_library(
    name = "torch_xla_test",
    srcs = ["torch_xla_test.cpp"],
    hdrs = ["torch_xla_test.h"],
    deps = [
        ":metrics_snapshot",
        "//third_party/xla_client:sys_util",
        "//third_party/xla_client:tf_logging",
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
        "//third_party/xla_client:computation_client",
        "//third_party/xla_client:debug_macros",
        "//third_party/xla_client:sys_util",
        "//torch_xla/csrc:tensor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_ir",
    srcs = ["test_ir.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_aten_xla_tensor",
    srcs = ["test_aten_xla_tensor.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//third_party/xla_client:metrics",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/compiler/xla:permutation_util",
    ],
)

ptxla_cc_library(
    name = "test_lazy",
    srcs = ["test_lazy.cpp"],
    deps = [
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
    ],
)

ptxla_cc_library(
    name = "test_op_by_op_executor",
    srcs = ["test_op_by_op_executor.cpp"],
    deps = [
        ":cpp_test_util",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_replication",
    srcs = ["test_replication.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//third_party/xla_client:computation_client",
        "//third_party/xla_client:debug_macros",
        "//third_party/xla_client:multi_wait",
        "//third_party/xla_client:thread_pool",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

ptxla_cc_library(
    name = "test_symint",
    srcs = ["test_symint.cpp"],
    deps = [
        ":cpp_test_util",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_tensor",
    srcs = ["test_tensor.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_xla_backend_intf",
    srcs = ["test_xla_backend_intf.cpp"],
    deps = [
        ":cpp_test_util",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
    ],
)

ptxla_cc_library(
    name = "test_xla_sharding",
    srcs = ["test_xla_sharding.cpp"],
    deps = [
        ":cpp_test_util",
        ":torch_xla_test",
        "//third_party/xla_client:env_vars",
        "//third_party/xla_client:sys_util",
        "//torch_xla/csrc:tensor",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
    ],
)

# Keep all the tests in the same binary as long as XRT is supported.
# Once PjRT is the only runtime, make each test library a test.
# The reason for this is that we want to run tests in parallel
# but XRT gRPC ports cannot be reused and are inhereted from the environment.
ptxla_cc_test(
    name = "main",
    size = "enormous",
    srcs = ["main.cpp"],
    deps = [
        ":test_aten_xla_tensor",
        ":test_ir",
        ":test_lazy",
        ":test_op_by_op_executor",
        ":test_replication",
        ":test_symint",
        ":test_tensor",
        ":test_xla_backend_intf",
        ":test_xla_sharding",
        "@com_google_googletest//:gtest_main",
    ],
)
