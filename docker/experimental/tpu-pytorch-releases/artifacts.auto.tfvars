nightly_package_version = "2.0"

# Built once a day from master.
nightly_builds = [
  { accelerator = "tpu" },
  {
    accelerator  = "cuda"
    cuda_version = "11.8"
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.7"
  }
]

# Built on push to specific tag.
versioned_builds = [
  {
    git_tag         = "v2.0.0"
    package_version = "2.0"
    accelerator     = "tpu"
  },
  {
    git_tag         = "v1.13.0"
    package_version = "1.13"
    accelerator     = "tpu"
  },
  {
    git_tag         = "v2.0.0"
    package_version = "2.0"
    accelerator     = "cuda"
    cuda_version    = "11.8"
  },
  {
    git_tag         = "v2.0.0"
    package_version = "2.0",
    accelerator     = "cuda"
    cuda_version    = "11.7"
  },
  {
    git_tag         = "v1.13.0"
    package_version = "1.13"
    accelerator     = "cuda"
    cuda_version    = "11.2"
  },
  # PyTorch doesn't build with Python 3.7.
  # {
  #   git_tag         = "v1.13.0"
  #   package_version = "1.13"
  #   accelerator     = "cuda"
  #   cuda_version    = "11.2"
  #   python_version  = "3.7"
  # },
  # {
  #   git_tag         = "v1.12.0"
  #   package_version = "1.12"
  #   accelerator     = "cuda"
  #   cuda_version    = "11.2"
  #   python_version  = "3.7"
  # },
]
