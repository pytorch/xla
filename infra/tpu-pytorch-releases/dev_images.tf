variable "dev_images" {
  type = list(object({
    accelerator    = string
    arch           = optional(string, "amd64")
    python_version = optional(string, "3.8")
    cuda_version   = optional(string, "11.8")

    # Additional tags on top of uniquely generated tag based on accelerator,
    # python and cuda versions.
    extra_tags = optional(list(string), [])
  }))
}

locals {
  dev_images_dict = {
    for di in var.dev_images :
    format("%s_%s",
      di.python_version,
      di.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", di.cuda_version)
    ) => di
  }
}

module "dev_images" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.dev_images_dict

  # Replace `.` and `_` with `-` as they're not allowed in trigger name.
  trigger_name = "dev-${replace(each.key, "/[_.]/", "-")}"

  trigger_on_push = {
    branch         = "master"
    included_files = ["infra/**"]
  }

  image_name = "development"
  image_tags = concat(each.value.extra_tags, [
    each.key,
    # Append _YYYYMMDD suffix to the dev image name.
    "${each.key}_$(date +%Y%m%d)",
  ])

  dockerfile = "development.Dockerfile"
  description = join(" ", [
    "Builds development:${each.key} image.",
    "Trigger managed by Terraform setup in infra/tpu-pytorch-releases/dev_images.tf.",
  ])

  build_args = {
    python_version = each.value.python_version
  }

  ansible_vars = {
    xla_git_rev     = "$COMMIT_SHA"
    pytorch_git_rev = "main"

    accelerator    = each.value.accelerator
    arch           = each.value.arch
    python_version = each.value.python_version
    cuda_version   = each.value.cuda_version
  }

  docker_repo_url = module.docker_registry.url
  worker_pool_id  = module.worker_pool.id
  timeout_minutes = 60
}
