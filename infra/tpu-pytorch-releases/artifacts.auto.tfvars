nightly_package_version = "2.2.0"

# Built once a day from master.
nightly_builds = [
  { accelerator = "tpu" },
  {
    accelerator    = "tpu"
    python_version = "3.10"
  },
  {
    accelerator  = "cuda"
    cuda_version = "12.1"
  },
  {
    accelerator  = "cuda"
    cuda_version = "12.0"
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.8"
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.8"
    python_version = "3.10"
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.7"
  }
]

# TODO: Remove this after the 2.1 release
xrt_versioned_builds = [
  {
    accelerator    = "tpu"
    python_version = "3.10"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0+xrt"
  },
  {
    accelerator  = "cuda"
    python_version = "3.10"
    cuda_version = "12.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0+xrt"
  },
]

# Built on push to specific tag.
versioned_builds = [
  # Remove libtpu from PyPI builds
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "tpu"
    bundle_libtpu   = "0"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "tpu"
    python_version  = "3.9"
    bundle_libtpu   = "0"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "tpu"
    python_version  = "3.10"
    bundle_libtpu   = "0"
  },
  # Bundle libtpu for Kaggle
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0+libtpu"
    accelerator     = "tpu"
    python_version  = "3.10"
    bundle_libtpu   = "1"
  },
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
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0",
    accelerator     = "cuda"
    cuda_version    = "12.0"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "cuda"
    cuda_version    = "11.8"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "cuda"
    cuda_version    = "12.1"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "cuda"
    cuda_version    = "11.8"
    python_version  = "3.10"
  },
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "cuda"
    cuda_version    = "12.1"
    python_version  = "3.10"
  },
  {
    git_tag         = "v2.0.0"
    package_version = "2.0"
    accelerator     = "cuda"
    cuda_version    = "11.8"
  },
  {
    git_tag         = "v2.0.0"
    package_version = "2.0"
    accelerator     = "cuda"
    cuda_version    = "11.8"
    python_version  = "3.10"
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
  {
    git_tag         = "v1.13.0"
    package_version = "1.13"
    accelerator     = "cuda"
    cuda_version    = "11.2"
    python_version  = "3.7"
  },
  {
    git_tag         = "v1.12.0"
    package_version = "1.12"
    accelerator     = "cuda"
    cuda_version    = "11.2"
    python_version  = "3.7"
  },
]
