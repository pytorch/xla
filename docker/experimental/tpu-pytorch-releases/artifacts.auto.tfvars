latest_package_version = "2.0"

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

versioned_builds = [
  {
    version     = "2.0"
    accelerator = "tpu"
  },
  {
    version     = "1.13"
    accelerator = "tpu"
  },
  {
    version      = "2.0"
    accelerator  = "cuda"
    cuda_version = "11.8"
  },
  {
    version      = "2.0",
    accelerator  = "cuda"
    cuda_version = "11.7"
  },
  {
    version      = "1.13"
    accelerator  = "cuda"
    cuda_version = "11.2"
  },
  {
    version        = "1.13"
    accelerator    = "cuda"
    cuda_version   = "11.2"
    python_version = "3.7"
  },
  {
    version        = "1.12"
    accelerator    = "cuda"
    cuda_version   = "11.2"
    python_version = "3.7"
  },
]
