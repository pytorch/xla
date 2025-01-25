# TODO(hanq): describe this package.
load(
    "//third_party/py/torch/google/bazel_rules/rules_python/python:defs.bzl",
    "py_test",
)

package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_test(
    name = "test_llama",
    srcs = [
        "llama_model.py",
        "test_llama.py",
    ],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch/google/_torx",
        "//third_party/py/torch/google/_torx/test:test_base",
    ],
)
