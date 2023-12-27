load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

copy_file(
  name = "plugin",
  src = "@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so",
  out = "pjrt_c_api_gpu_plugin.so",
)

py_library(
  name = "torch_xla_cuda_plugin",
  srcs = ["__init__.py"],
)
