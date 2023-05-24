load("//bazel:rules_def.bzl", "ptxla_py_test")

ptxla_py_test(
    name = "test_bridge",
    srcs = ["test_bridge.py"],
    accelerators = ["cpu"],
    modes = ["default"],
)

ptxla_py_test(
    name = "test_dynamo_integrations_util",
    srcs = ["test_dynamo_integrations_util.py"],
    accelerators = ["cpu"],
    modes = ["default"],
)

ptxla_py_test(
    name = "test_num_output",
    srcs = ["test_num_output.py"],
    accelerators = ["cpu"],
    modes = ["default"],
)

ptxla_py_test(
    name = "test_dynamo",
    srcs = ["test_dynamo.py"],
    accelerators = ["cpu"],
    modes = ["default"],
)

ptxla_py_test(
    name = "test_dynamo_graph_dump",
    srcs = ["test_dynamo_graph_dump.py"],
    accelerators = ["cpu"],
    modes = ["save_tensor_file"],
)
