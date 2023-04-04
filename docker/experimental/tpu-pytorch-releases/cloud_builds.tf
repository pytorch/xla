variable "nightly_package_version" {
  type = string
}

variable "nightly_builds" {
  type = list(
    object({
      accelerator    = string
      cuda_version   = optional(string, "11.8")
      python_version = optional(string, "3.8")
      arch           = optional(string, "amd64")
    })
  )

  default = []
}

variable "versioned_builds" {
  type = list(
    object({
      git_tag         = string
      package_version = string
      accelerator     = string
      python_version  = optional(string, "3.8")
      cuda_version    = optional(string, "11.8")
      arch            = optional(string, "amd64")
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

  ansible_vars = merge(each.value, {
    package_version = var.nightly_package_version
    nightly_release = true
    pytorch_git_rev = "master"
    xla_git_rev     = "master"
  })

  # TODO: Change this branch to master. Currently the dev branch contains
  # Ansible with deps for older versions of CUDA (see cuda_deps.yaml).
  ansible_branch      = "mlewko/terraform-follow-up"
  trigger_on_schedule = { schedule = "0 0 * * *", branch = "master" }

  trigger_name = "nightly-${replace(each.key, "/[_.]/", "-")}"
  image_name   = "xla"
  image_tags = [
    "nightly_${each.key}",
    # Append _YYYYMMDD suffix to nightly image name.
    "nightly_${each.key}_$(date +%Y%m%d)",
  ]

  description = join(" ", [
    "Builds nightly xla:nightly_${each.key}' ${
      each.value.accelerator == "tpu"
      ? "TPU"
      : format("CUDA %s", each.value.cuda_version)
    } docker image and corresponding wheels for PyTorch/XLA.",
    "Trigger managed by Terraform setup in",
    "docker/experimental/tpu-pytorch-releases/cloud_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${
    each.value.accelerator == "tpu"
    ? "tpuvm"
    : "cuda/${each.value.cuda_version}"
  }"
  wheels_srcs = ["/dist/*.whl"]
  build_args = {
    python_version = each.value.python_version
  }

  scheduler_account_email = module.scheduler_account.email
  worker_pool_id          = module.worker_pool.id
  docker_repo_url         = module.docker_registry.url
}

module "versioned_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.versioned_builds_dict

  ansible_vars = merge(each.value, {
    pytorch_git_rev = each.value.git_tag
    xla_git_rev     = each.value.git_tag
  })

  # TODO: Change this branch to master. Currently the dev branch contains
  # Ansible with deps for older versions of CUDA (see cuda_deps.yaml).
  ansible_branch  = "mlewko/terraform-follow-up"
  trigger_on_push = { tag = each.value.git_tag }

  trigger_name = replace(each.key, "/[_.]/", "-")
  image_name   = "xla"
  image_tags   = [each.key]

  description = join(" ", [
    "Builds official xla:${each.key}' ${
      each.value.accelerator == "tpu"
      ? "TPU"
      : format("CUDA %s", each.value.cuda_version)
    } docker image and corresponding wheels for PyTorch/XLA.",
    "Trigger managed by Terraform setup in",
    "docker/experimental/tpu-pytorch-releases/cloud_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${
    each.value.accelerator == "tpu"
    ? "tpuvm"
    : "cuda/${each.value.cuda_version}"
  }"
  wheels_srcs = ["/dist/*.whl"]
  build_args  = each.value

  worker_pool_id  = module.worker_pool.id
  docker_repo_url = module.docker_registry.url
}

variable "versioned_builds" {
  type = list(
    object({
      git_tag         = string
      package_version = optional(string, "")
      python_version  = optional(string, "3.8")
      arch            = optional(string, "amd64")
      accelerator     = string
      python_version  = optional(string, "3.8")
      cuda_version    = optional(string, "11.8")
      arch            = optional(string, "amd64")
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
  # TODO: Change this branch to master. Currently the dev branch contains
  # Ansible with deps for older versions of CUDA (see cuda_deps.yaml).
  ansible_branch      = "mlewko/terraform-follow-up"
  trigger_on_schedule = { schedule = "0 0 * * *", branch = "master" }

  trigger_name = "nightly-${replace(each.key, "/[_.]/", "-")}"
  image_name   = "xla"
  image_tags = [
    "nightly_${each.key}",
    # Append _YYYYMMDD suffix to nightly image name.
    "nightly_${each.key}_$(date +%Y%m%d)",
  ]

  description = join(" ", [
    "Builds nightly xla:nightly_${each.key}' ${
      each.value.accelerator == "tpu"
      ? "TPU"
      : format("CUDA %s", each.value.cuda_version)
    } docker image and corresponding wheels for PyTorch/XLA.",
    "Trigger managed by Terraform setup in",
    "docker/experimental/tpu-pytorch-releases/cloud_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${
    each.value.accelerator == "tpu"
    ? "tpuvm"
    : "cuda/${each.value.cuda_version}"
  }"
  wheels_srcs = ["/dist/*.whl"]
  build_args = {
    python_version = each.value.python_version
  }

  scheduler_account_email = module.scheduler_account.email
  worker_pool_id          = module.worker_pool.id
  docker_repo_url         = module.docker_registry.url
}

module "versioned_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.versioned_builds_dict

  ansible_vars = merge(each.value, {
    pytorch_git_rev = each.value.git_tag
    xla_git_rev     = each.value.git_tag
  })

  # TODO: Change this branch to master. Currently the dev branch contains
  # Ansible with deps for older versions of CUDA (see cuda_deps.yaml).
  ansible_branch  = "mlewko/terraform-follow-up"
  trigger_on_push = { tag = each.value.git_tag }

  trigger_name = replace(each.key, "/[_.]/", "-")
  image_name   = "xla"
  image_tags   = [each.key]

  description = join(" ", [
    "Builds official xla:${each.key}' ${
      each.value.accelerator == "tpu"
      ? "TPU"
      : format("CUDA %s", each.value.cuda_version)
    } docker image and corresponding wheels for PyTorch/XLA.",
    "Trigger managed by Terraform setup in",
    "docker/experimental/tpu-pytorch-releases/cloud_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${
    each.value.accelerator == "tpu"
    ? "tpuvm"
    : "cuda/${each.value.cuda_version}"
  }"
  wheels_srcs = ["/dist/*.whl"]
  build_args  = each.value

  worker_pool_id  = module.worker_pool.id
  docker_repo_url = module.docker_registry.url
}
