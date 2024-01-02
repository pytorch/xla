nightly_package_version = "2.3.0"

# Built once a day from master.
nightly_builds = [
  { accelerator = "tpu" },
  {
    accelerator    = "tpu"
    python_version = "3.9"
  },
  {
    accelerator    = "tpu"
    python_version = "3.10"
  },
  {
    accelerator    = "tpu"
    python_version = "3.11"
  },
  {
    accelerator  = "cuda"
    cuda_version = "12.1"
  },
]

# Built on push to specific tag.
versioned_builds = [
  # Remove libtpu from PyPI builds
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    pytorch_git_rev = "v2.2.0-rc6"
    accelerator     = "tpu"
    bundle_libtpu   = "0"
  },
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    pytorch_git_rev = "v2.2.0-rc6"
    accelerator     = "tpu"
    python_version  = "3.9"
    bundle_libtpu   = "0"
  },
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    pytorch_git_rev = "v2.2.0-rc6"
    accelerator     = "tpu"
    python_version  = "3.10"
    bundle_libtpu   = "0"
  },
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    pytorch_git_rev = "v2.2.0-rc6"
    accelerator     = "tpu"
    python_version  = "3.11"
    bundle_libtpu   = "0"
  },
  # Bundle libtpu for Kaggle
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6+libtpu"
    pytorch_git_rev = "v2.2.0-rc6"
    accelerator     = "tpu"
    python_version  = "3.10"
    bundle_libtpu   = "1"
  },
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    accelerator     = "cuda"
    cuda_version    = "12.1"
  },
  {
    git_tag         = "v2.2.0-rc6"
    package_version = "2.2.0rc6"
    accelerator     = "cuda"
    cuda_version    = "12.1"
    python_version  = "3.10"
  },
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
  {
    git_tag         = "v2.1.0"
    pytorch_git_rev = "v2.1.0"
    package_version = "2.1.0"
    accelerator     = "tpu"
    python_version  = "3.11"
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
