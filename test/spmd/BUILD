load("//bazel:rules_def.bzl", "ptxla_py_test")

ptxla_py_test(
    name = "test_xla_sharding",
    srcs = ["test_xla_sharding.py"],
    accelerators = ["cpu"],
    modes = ["default"],
    tags = ["xla_op"],
)

ptxla_py_test(
    name = "test_xla_virtual_device",
    srcs = ["test_xla_virtual_device.py"],
    accelerators = ["cpu"],
    modes = ["default"],
    tags = ["xla_op"],
)
