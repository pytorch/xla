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
