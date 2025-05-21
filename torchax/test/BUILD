# TODO(hanq): describe this package.

load(
    "//third_party/py/torch/google/bazel_rules/rules_python/python:defs.bzl",
    "py_library",
    "py_test",
)

package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "base_test_util",
    srcs = ["base_test_util.py"],
    deps = [
        "//testing/pybase",
    ],
)

py_test(
    name = "test_core_aten_ops",
    srcs = ["test_core_aten_ops.py"],
    deps = [
        ":base_test_util",
        "//third_party/py/absl:app",
        "//third_party/py/torch/google/_torx",
    ],
)
