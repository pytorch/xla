module "dev_image" {
  source = "../terraform_modules/xla_docker_build"

  trigger_name = "dev-image"

  ansible_branch = "master"
  trigger_on_push = {
    branch        = "master"
    include_files = ["infra/**"]
  }

  image_name = "development"
  image_tags = [
    "tpu",
    # Append _YYYYMMDD suffix to nightly image name.
    "tpu_$(date +%Y%m%d)",
  ]
  dockerfile = "development.Dockerfile"
  description = join(" ", [
    "Build development image with TPU support.",
    "Trigger managed by Terraform setup in",
    "infra/tpu-pytorch/cloud_builds.tf.",
  ])

  build_args = {
    python_version = "3.8"
    debian_version = "buster"
  }

  ansible_vars = {
    arch        = "amd64"
    accelerator = "tpu"
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
