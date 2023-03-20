# Docker images.
docker_images = [
  {
    image = "development"
    description = "Build development image with TPU support."
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
  {
    name = "xla-nightly-38-cuda11-8"
    image = "xla"
    description = "Build nightly image with CUDA support"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version="3.8"
      arch="amd64"
      accelerator="cuda"
      cuda_version="11.8"
    }
    image_tags = [
      "nightly_3.8_cuda11.8",
      "nightly_3.8_cuda11.8_$(date +%Y%m%d)",
    ]
    wheels = ["/wheels/*.whl"]
  },
  {
    name = "xla-nightly-38-tpu"
    image = "xla"
    description = "Build nightly image with TPU support"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version="3.8"
      arch="amd64"
      accelerator="tpu"
    }
    image_tags = [
      "nightly_3.8_tpuvm",
      "nightly_3.8_$(date +%Y%m%d)",
    ]
    wheels = ["/wheels/*.whl"]
  },
  {
    name = "xla-1-13-38-tpu"
    image = "xla"
    description = "Build v1.13.0 image with TPU support"
    # Don't use this tag here, since the repositiory at version v1.13.0
    # doesn't contain ansible setup.
    # git_tag = "v1.13.0"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version="3.8"
      arch="amd64"
      accelerator="tpu"
      # Fetch sources at the desired tag.
      pytorch_git_rev="v1.13.0"
      xla_git_rev="v1.13.0"
    }
    image_tags = [
      "r1.13_3.8_tpuvm",
    ]
    wheels = ["/wheels/*.whl"]
  },
  {
    name = "xla-1-12-38-tpu"
    image = "xla"
    description = "Build v1.12.0 image with TPU support"
    # git_tag = "v1.12.0"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version="3.8"
      arch="amd64"
      accelerator="tpu"
      pytorch_git_rev="v1.13.0"
      xla_git_rev="v1.13.0"
    }
    image_tags = [
      "r1.12_3.8_tpuvm",
    ]
    wheels = ["/wheels/*.whl"]
  },
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