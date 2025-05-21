load(
    "@xla//xla:xla.bzl",
    "xla_cc_binary",
)

cc_library(
    name = "test_cpu_plugin",
    srcs = ["test_cpu_plugin.cpp"],
    hdrs = ["test_cpu_plugin.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_cpu_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_c_api_cpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    deps = [
        ":pjrt_c_api_cpu_version_script.lds",
        ":test_cpu_plugin",
    ],
)
