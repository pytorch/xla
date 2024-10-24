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
      bundle_libtpu  = optional(string, "0")
      cxx11_abi      = optional(string, "0")
    })
  )

  default = []
}

// TODO: Remove this after the 2.1 release
variable "xrt_versioned_builds" {
  type = list(
    object({
      package_version = string
      accelerator    = string
      pytorch_git_rev = optional(string, "")
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
      # Fetch PyTorch at a given revision (e.g. git tag), otherwise use git_tag.
      pytorch_git_rev = optional(string, "")
      python_version  = optional(string, "3.8")
      cuda_version    = optional(string, "11.8")
      arch            = optional(string, "amd64")
      bundle_libtpu   = optional(string, "0")
      cxx11_abi       = optional(string, "0")
    })
  )

  default = []
}

locals {
  nightly_builds_dict = {
    for b in var.nightly_builds :
    format("%s_%s%s",
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version),
      b.cxx11_abi == "1" ? "_cxx11" : ""
    ) => b
  }

  // TODO: Remove this after the 2.1 release
  xrt_versioned_builds_dict = {
    for b in var.xrt_versioned_builds :
    format("r%s_%s_%s",
      replace(b.package_version, "+", "_"),
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version)
    ) => b
  }

  versioned_builds_dict = {
    for b in var.versioned_builds :
    format("r%s_%s_%s%s",
      replace(b.package_version, "+", "_"),
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version),
      b.cxx11_abi == "1" ? "_cxx11" : ""
    ) => b
  }
}

module "nightly_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.nightly_builds_dict

  ansible_vars = merge(each.value, {
    package_version = var.nightly_package_version
    nightly_release = true
    pytorch_git_rev = "main"
    xla_git_rev     = "$COMMIT_SHA"
  })

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
    "infra/tpu-pytorch-releases/artifacts_builds.tf."
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

// TODO: Remove this after the 2.1 release
module "xrt_versioned_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.xrt_versioned_builds_dict

  ansible_vars = merge(each.value, {
    xla_git_rev     = "$COMMIT_SHA",
    cxx11_abi       = each.value.cxx11_abi
  })

  trigger_on_schedule = { schedule = "0 0 * * *", branch = "xrt" }

  trigger_name = replace(each.key, "/[_.]/", "-")
  image_name   = "xla"
  image_tags = [
    each.key,
    # Append _YYYYMMDD suffix to nightly image name.
    "${each.key}_$(date +%Y%m%d)",
  ]

  description = join(" ", [
    "Builds nightly xla:${each.key}' ${
      each.value.accelerator == "tpu"
      ? "TPU"
      : format("CUDA %s", each.value.cuda_version)
    } docker image and corresponding wheels for PyTorch/XLA.",
    "Trigger managed by Terraform setup in",
    "infra/tpu-pytorch-releases/artifacts_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/xrt/${
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
    // Override `pytorch_git_rev` set in each value of `versioned_builds_dict`
    // if it's left empty.
    pytorch_git_rev = coalesce(each.value.pytorch_git_rev, each.value.git_tag)
    xla_git_rev     = each.value.git_tag,
    cxx11_abi       = each.value.cxx11_abi
  })

  # Use Ansible setup from master branch for versioned release, because source
  # code at older version doesn't contain Ansible setup.
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
    "infra/tpu-pytorch-releases/artifacts_builds.tf."
  ])

  wheels_dest = "${module.releases_storage_bucket.url}/wheels/${
    each.value.accelerator == "tpu"
    ? "tpuvm"
    : "cuda/${each.value.cuda_version}"
  }"
  wheels_srcs = ["/dist/*.whl"]
  # Pass docker build args to infra/ansible/Dockerfile, other than `ansible_vars`.
  build_args = {
    python_version = each.value.python_version
  }

  worker_pool_id  = module.worker_pool.id
  docker_repo_url = module.docker_registry.url
}
