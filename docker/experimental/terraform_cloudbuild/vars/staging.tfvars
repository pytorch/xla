# Docker images.
docker_images = [
  {
    image = "development"
    description = "Build development image with TPU support."
    branch = "mlewko/cloudbuild"
    dir = "docker/experimental/ansible"
    dockerfile = "development.Dockerfile"
    trigger_files = [
      "docker/experimental/ansible/**",
      "docker/experimental/terraform_cloudbuild/**",
    ]
    build_args = {
      python_version="3.8"
      arch="amd64"
      accelerator="tpu"
    }
    image_tags = [
      "tpu_amd64",
      "tpu_amd64_$(date +%Y%m%d)",
    ]
  },
  # {
  #   image = "xla"
  #   description = "Build release images"
  #   branch = "mlewko/cloudbuild"
  #   dir = "docker/experimental/ansible"
  #   dockerfile = "development.Dockerfile"
  #   image_tags = [
  #     "latest",
  #     "$(date +%Y%m%d)",
  #   ]
  # }
]

# Variables for the staging environment.

project_id = "tpu-pytorch"
public_docker_repo = {
  id = "docker-public-staging"
}
worker_pool = {
  name = "worker-pool-staging"
  machine_type = "e2-highcpu-32"
}
storage_bucket_suffix = "-staging"
build_runner_account_id_suffix = "-staging"
triggers_suffix = "-staging"