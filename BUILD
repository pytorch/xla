load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
    generate_hashes = True,
)

cc_binary(
    name = "_XLAC.so",
    copts = [
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_XLAC",
        "-fopenmp",
        "-fPIC",
        "-fwrapv",
    ],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/torch_xla/lib",  # for libtpu
        "-Wl,-soname,_XLAC.so",
        "-lstdc++fs",  # For std::filesystem
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//torch_xla/csrc:init_python_bindings",
        "@torch//:headers",
        "@torch//:libc10",
        "@torch//:libtorch",
        "@torch//:libtorch_cpu",
        "@torch//:libtorch_python",
    ],
)

test_suite(
    name = "cpp_tests",
    # testonly = True,
    tests = [
        "//test/cpp:test_aten_xla_tensor_1",
        "//test/cpp:test_aten_xla_tensor_2",
        "//test/cpp:test_aten_xla_tensor_3",
        "//test/cpp:test_aten_xla_tensor_4",
        "//test/cpp:test_aten_xla_tensor_5",
        "//test/cpp:test_aten_xla_tensor_6",
        "//test/cpp:test_debug_macros",
        "//test/cpp:test_device",
        "//test/cpp:test_ir",
        "//test/cpp:test_lazy",
        "//test/cpp:test_replication",
        "//test/cpp:test_runtime",
        "//test/cpp:test_status_dont_show_cpp_stacktraces",
        "//test/cpp:test_status_show_cpp_stacktraces",
        "//test/cpp:test_tensor",
        "//test/cpp:test_xla_generator",
        "//test/cpp:test_xla_sharding",
        "//torch_xla/csrc/runtime:pjrt_computation_client_test",
        # "//torch_xla/csrc/runtime:ifrt_computation_client_test",
    ],
)
