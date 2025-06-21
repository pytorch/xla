########## Begin section for release and nightly ########
# Define common configuration parameters for 2.7 release and nightly
locals {
  tpu_python_versions = ["3.9", "3.10", "3.11", "3.12"]
  release_git_tag         = "v2.7.0-rc5"
  release_package_version = "2.7.0-rc5"
  release_pytorch_git_rev = "v2.7.0-rc10"
  nightly_package_version = "2.8.0"
  cuda_versions = {
    "nightly": [],
    "r2.7": ["12.1", "12.6"] # Note: PyTorch 2.7 release supports 11.8, 12.6, 12.8
  }

  # Built once a day from master
  generated_nightly_builds = concat(
    # TPU builds
    [
      for py_ver in local.tpu_python_versions : {
        accelerator    = "tpu"
        python_version = py_ver
        bundle_libtpu  = "0"
        cxx11_abi      = "1"
      }
    ],
    # CUDA builds
    [
      for pair in setproduct(local.tpu_python_versions, local.cuda_versions["nightly"]) : {
        accelerator     = "cuda"
        cuda_version    = pair[1]
        python_version  = pair[0]
        bundle_libtpu   = "0"
        cxx11_abi       = "1"
      }
    ]
  )

  # Built on push to specific tag.
  generated_versioned_builds = concat(
    # Regular TPU builds (non-libtpu, C++11 ABI)
    [
      for py_ver in local.tpu_python_versions : {
        git_tag         = local.release_git_tag
        package_version = local.release_package_version
        pytorch_git_rev = local.release_pytorch_git_rev
        accelerator     = "tpu"
        python_version  = py_ver
        bundle_libtpu   = "0"
      }
    ],

    # Special Kaggle build with libtpu
    [
      {
        git_tag         = local.release_git_tag
        package_version = "${local.release_package_version}+libtpu"
        pytorch_git_rev = local.release_pytorch_git_rev
        accelerator     = "tpu"
        python_version  = "3.10"
        bundle_libtpu   = "1"
      }
    ],

    # cuda build for latest release
    [
    for pair in setproduct(local.tpu_python_versions, local.cuda_versions["r2.7"]) : {
      git_tag         = local.release_git_tag
      package_version = local.release_package_version
      pytorch_git_rev = local.release_pytorch_git_rev
      accelerator     = "cuda"
      cuda_version    = pair[1]
      python_version  = pair[0]
      bundle_libtpu   = "0"
    }
    ]
  )
  versioned_builds = concat(local.generated_versioned_builds, var.manual_versioned_builds)
  nightly_builds = concat(local.generated_nightly_builds, var.manual_nightly_builds)
}


########## End section for release and nightly ########


# Add this variable declaration
variable "manual_versioned_builds" {
  description = "Historical build configurations provided via tfvars"
  type = list(
    object({
      git_tag         = string
      package_version = string
      accelerator     = string
      pytorch_git_rev = optional(string, "")
      python_version  = optional(string, "3.8")
      cuda_version    = optional(string, "11.8")
      arch            = optional(string, "amd64")
      bundle_libtpu   = optional(string, "0")
      cxx11_abi       = optional(string, "1")
    })
  )
  default = []
}


variable "manual_nightly_builds" {
  type = list(
    object({
      accelerator    = string
      cuda_version   = optional(string, "11.8")
      python_version = optional(string, "3.8")
      arch           = optional(string, "amd64")
      bundle_libtpu  = optional(string, "0")
      cxx11_abi      = optional(string, "1")
    })
  )

  default = []
}

locals {
  nightly_builds_dict = {
    for b in local.nightly_builds :
    format("%s_%s%s%s",
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version),
      try(b.cxx11_abi == "0", false) ? "_precxx11" : "",
      try(b.bundle_libtpu == "1", false) ? "_libtpu" : ""
    ) => b
  }


  versioned_builds_dict = {
    for b in local.versioned_builds :
    format("r%s_%s_%s%s%s",
      replace(b.package_version, "+", "_"),
      b.python_version,
      b.accelerator == "tpu" ? "tpuvm" : format("cuda_%s", b.cuda_version),
      try(b.cxx11_abi == "0", false) ? "_precxx11" : "",
      try(b.bundle_libtpu == "1", false) ? "_libtpu" : ""
    ) => b
  }
}

module "nightly_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.nightly_builds_dict

  ansible_vars = merge(each.value, {
    package_version = local.nightly_package_version
    nightly_release = true
    pytorch_git_rev = "main"
    xla_git_rev     = "$COMMIT_SHA"
    arch            = lookup(each.value, "arch", "amd64")
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


module "versioned_builds" {
  source   = "../terraform_modules/xla_docker_build"
  for_each = local.versioned_builds_dict

  ansible_vars = merge(each.value, {
    // Override `pytorch_git_rev` set in each value of `versioned_builds_dict`
    // if it's left empty.
    pytorch_git_rev = coalesce(each.value.pytorch_git_rev, each.value.git_tag)
    xla_git_rev     = each.value.git_tag,
    cxx11_abi       = lookup(each.value, "cxx11_abi", "1")
    arch            = lookup(each.value, "arch", "amd64")
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
