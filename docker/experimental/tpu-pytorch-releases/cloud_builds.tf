module "nightly_builds" {
  source   = "../terraform_modules/build_trigger"
  for_each = nightly_builds_dict

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
  nightly_builds_dict = {
    for di in var.nightly_builds :
    format("%s_%s",
      di.python_version,
      di.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", di.cuda_version)
    ) => di
  }
}

variable "nightly_builds" {
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
