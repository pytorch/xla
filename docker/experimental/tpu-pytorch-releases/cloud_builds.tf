module "nightly_docker_images" {
  source   = "../terraform_modules/docker_image_build"
  for_each = nightly_docker_images_map

  scheduler_account_email = scheduler_account.email

  sources_git_rev = "master"
  ansible_git_rev = "master"

  image_name = each.value.image_name
  image_tags = [
    "nightly_${each.key}",
    # Append _YYYYMMDD suffix to nightly image name.
    "nightly_${each.key}_$(date +%Y%m%d)",
  ]

  wheels_dest = "${releases_storage_bucket.url}/tpuvm"
  wheels_srcs = [
    "/src/pytorch/dist",
    "/src/pytorch/xla/dist",
  ]

  arch     = "amd64"
  schedule = "0 0 * * *"
}

locals {
  nightly_docker_images_map = {
    for di in var.nightly_docker_images :
    format("%s_%s",
      di.python_version,
      di.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", di.cuda_version)
    ) => di
  }
}

variable "nightly_docker_images" {
  type = list(
    object({
      image_name     = optional(string, "xla")
      python_version = optional(string, "3.8")
      arch           = optional(string, "amd64")
      accelerator    = string
      cuda_version   = optional(string, "")
    })
  )

  default = []
}
