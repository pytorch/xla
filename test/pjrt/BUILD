load("//bazel:rules_def.bzl", "ptxla_py_test")

ptxla_py_test(
    name = "test_ddp",
    srcs = ["test_ddp.py"],
    accelerators = ["cpu"],
    modes = ["dynamic"],
)

ptxla_py_test(
    name = "test_experimental_pjrt",
    srcs = ["test_experimental_pjrt.py"],
    accelerators = ["cpu"],
    modes = ["dynamic"],
)

ptxla_py_test(
    name = "test_experimental_tpu",
    srcs = ["test_experimental_tpu.py"],
    accelerators = ["cpu"],
    modes = ["dynamic"],
)

ptxla_py_test(
    name = "test_mesh_service",
    srcs = ["test_mesh_service.py"],
    accelerators = ["cpu"],
    modes = ["dynamic"],
)
