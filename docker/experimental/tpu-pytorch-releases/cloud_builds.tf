variable "nightly_package_version" {
  type = string
}

variable "nightly_builds" {
  type = list(
    object({
      python_version = optional(string, "3.8")
      arch           = optional(string, "amd64")
      accelerator    = string
      cuda_version   = optional(string, "")
    })
  )

  default = []
}

variable "versioned_builds" {
  type = list(
    object({
      git_tag         = string
      package_version = optional(string, "")
      python_version  = optional(string, "3.8")
      arch            = optional(string, "amd64")
      accelerator     = string
      cuda_version    = optional(string, "")
    })
  )

  default = []
}

locals {
  nightly_builds_dict = {
    for b in var.nightly_builds :
    format("%s_%s",
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version)
    ) => b
  }

  versioned_builds_dict = {
    for b in var.versioned_builds :
    format("r%s_%s_%s",
      b.package_version,
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version)
    ) => b
  }
}

module "nightly_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.nightly_builds_dict

  sources_git_rev = "master"
  ansible_branch  = "master"

  trigger_name = "nightly_${each.key}"
  image_name   = "xla"
  image_tags = [
    "nightly_${each.key}",
    # Append _YYYYMMDD suffix to nightly image name.
    "nightly_${each.key}_$(date +%Y%m%d)",
  ]

  description = format(
    "Builds nightly 'xla:nightly_%s' %s docker image and corresponding wheels for PyTorch/XLA. Configured in Terraform.",
    each.key,
    each.value.accelerator == "tpu" ? "TPU" : format("CUDA %s", each.value.cuda_version)
  )

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${each.key}"
  wheels_srcs = ["/dist/*.whl"]
  build_args  = merge(each.value, { package_version = var.nightly_package_version })

  schedule                = "0 0 * * *"
  scheduler_account_email = module.scheduler_account.email
  worker_pool_id          = module.worker_pool.id
  docker_repo_url         = module.docker_registry.url
}

module "versioned_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.versioned_builds_dict

  sources_git_rev = each.value.git_tag
  ansible_branch = "master"

  trigger_name = each.key
  image_name   = "xla"
  image_tags   = [each.key]

  description = format(
    "Builds official 'xla:%s' %s docker image and corresponding wheels for PyTorch/XLA. Configured in Terraform.",
    each.key,
    each.value.accelerator == "tpu" ? "TPU" : format("CUDA %s", each.value.cuda_version)
  )

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${each.key}"
  wheels_srcs = ["/dist/*.whl"]
  build_args  = each.value

  worker_pool_id  = module.worker_pool.id
  docker_repo_url = module.docker_registry.url
}
