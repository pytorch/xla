# Docker imagess.
docker_images = [
  {
    image = "development_tpu"
    description = "Build development image with TPU support."
    branch = "mlewko/cloudbuild"
    trigger_files = [
      "docker/experimental/ansible/**",
      "docker/experimental/terraform_cloudbuild/**",
    ]
    dir = "docker/experimental/ansible"
    dockerfile = "development.Dockerfile"
    tags = [
      "latest",
      "$(date +%Y%m%d)",
    ]
  }
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