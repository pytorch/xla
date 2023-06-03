load("//bazel:rules_def.bzl", "ptxla_py_test")

ptxla_py_test(
    name = "test_dynamic_shapes",
    srcs = ["test_dynamic_shapes.py"],
    modes = ["dynamic"],
    tags = ["xla_op"],
)

ptxla_py_test(
    name = "test_dynamic_shape_models",
    srcs = ["test_dynamic_shape_models.py"],
    modes = ["dynamic"],
    tags = ["xla_op"],
)
