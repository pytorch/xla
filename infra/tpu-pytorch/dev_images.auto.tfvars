dev_images = [
  {
    accelerator = "tpu"
    extra_tags  = ["tpu"]
  },
  {
    accelerator  = "cuda"
    cuda_version = "11.8"
    extra_tags   = ["cuda"]
  }
]
