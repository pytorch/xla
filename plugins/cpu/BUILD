# PJRT GPU plugin. Can be configured to be built for CUDA or ROCM.
load(
    "@xla//xla:xla.bzl",
    "xla_cc_binary",
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
        "@xla//xla/pjrt/c:pjrt_c_api_cpu"
    ],
)
