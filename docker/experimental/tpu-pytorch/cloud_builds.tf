module "dev_image" {
  source = "../terraform_modules/xla_docker_build"

  trigger_name = "dev-image"

  ansible_branch = "mlewko/terraform-follow-up"
  trigger_on_push = {
    branch = "master"
    include_files = [
      "docker/experimental/ansible/**",
      "docker/experimental/terraform/**",
      "docker/experimental/terraform_modules/**",
      "docker/experimental/tpu-pytorch/**",
      "docker/experimental/tpu-pytorch-releases/**",
    ]
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
    "docker/experimental/tpu-pytorch/cloud_builds.tf.",
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
