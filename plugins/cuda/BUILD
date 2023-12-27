genrule(
  name = "wheel",
  srcs = [
    "pyproject.toml",
    "//plugins/cuda/torch_xla_cuda_plugin",
    "//plugins/cuda/torch_xla_cuda_plugin:plugin",
  ],
  outs = ["torch_xla_cuda_plugin-0.0.1-py3-none-any.whl"],
  cmd = "pip wheel plugins/cuda -w $(@D)",
)
