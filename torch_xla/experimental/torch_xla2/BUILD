load("//third_party/py/torch/google/bazel_rules/rules_python/python:defs.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = [
        "//learning/pytorch:__subpackages__",
        "//third_party/py/torch/google:__subpackages__",
    ],
    licenses = ["notice"],
)

# Note that all the targets shouldn't be included by torch:google, to avoid the large dependency
# between torch and tf. User who would like to use these utils should explicit import them.
py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":ops_registry",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "ops_registry",
    srcs = ["ops_registry.py"],
    deps = [
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "tensor",
    srcs = ["tensor.py"],
    deps = [
        ":ops_registry",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":ops_registry",
        ":tensor",
        "//third_party/py/jax",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "tf_integration",
    srcs = ["tf_integration.py"],
    deps = [
        ":export",
        "//third_party/py/jax/experimental/jax2tf",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "_torx",
    srcs = ["__init__.py"],
    deps = [
        ":export",
        ":ops",
        ":ops_registry",
        ":tensor",
        "//third_party/py/torch:pytorch",
    ],
)
