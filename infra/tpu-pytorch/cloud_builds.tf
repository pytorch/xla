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

  ansible_branch = "master"
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
    "Trigger managed by Terraform setup in infra/tpu-pytorch/cloud_builds.tf.",
  ])

  build_args = {
    python_version = each.value.python_version
    debian_version = "buster"
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
  location        = "global"
}

# This trigger should be removed once bazel is merged.
module "bazel_build" {
  source = "../terraform_modules/xla_docker_build"

  ansible_vars = {
    package_version = "2.0"
    pytorch_git_rev = "main"
    xla_git_rev     = "bazel-torchxla"
    python_version  = "3.8"
    arch            = "amd64"
    accelerator     = "tpu"
  }

  ansible_branch  = "bazel-torchxla"
  trigger_on_push = { branch = "bazel-torchxla" }

  trigger_name = "xla-bazel"
  image_name   = "xla"
  image_tags   = ["bazel"]

  description = join(" ", [
    "Bazel build of PyTorch/XLA (bazel-torchxla branch).",
    "Trigger managed by Terraform setup in",
    "infra/tpu-pytorch/cloud_builds.tf."
  ])

  # TODO: This bucket was created for testing and should be removed.
  wheels_dest = "gs://tpu-pytorch-wheels-public/bazel/tpuvm"
  wheels_srcs = ["/dist/*.whl"]

  worker_pool_id  = module.worker_pool.id
  docker_repo_url = module.docker_registry.url

  location = "global"
}
