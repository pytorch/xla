exports_files(["xla_native_functions.yaml"])

# Requires `torchgen` locally, via pip `torch` package or local builds.
py_binary(
    name = "lazy_tensor_generator",
    srcs = ["lazy_tensor_generator.py"],
    data = [
        "//torch_xla/csrc:aten_xla_type.cpp",
        "@torch//:torchgen_deps",
    ],
    tags = [
        "local",
        "no-remote-exec",
    ],
)

sh_binary(
    name = "fix_includes",
    srcs = ["fix_includes.sh"],
)
