dev_images = [
  {
    accelerator = "tpu"
    extra_tags  = ["tpu"]
    python_version = "3.10"
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.8"
    extra_tags   = ["cuda"]
  },
  {
    accelerator  = "cuda"
    cuda_version = "12.1"
    extra_tags   = ["cuda"]
  },
  {
    accelerator  = "cuda"
    cuda_version = "12.3"
    extra_tags   = ["cuda"]
  }
]
