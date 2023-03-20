# Each docker images is translated into a single build trigger.
docker_images = [
  {
    image       = "development"
    description = "Build development image with TPU support."
    dockerfile  = "development.Dockerfile"
    trigger_files = [
      "docker/experimental/ansible/**",
      "docker/experimental/terraform_cloudbuild/**",
    ]
    build_args = {
      python_version = "3.8"
      arch           = "amd64"
      accelerator    = "tpu"
    }
    image_tags = [
      "tpu_amd64",
      "tpu_amd64_$(date +%Y%m%d)",
    ]
  },
  {
    trigger_name     = "xla-nightly-38-cuda11-8"
    image            = "xla"
    description      = "Build nightly image with CUDA support"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version  = "3.8"
      arch            = "amd64"
      accelerator     = "cuda"
      cuda_version    = "11.8"
      package_version = "2.0"
    }
    image_tags = [
      "nightly_3.8_cuda11.8",
      "nightly_3.8_cuda11.8_$(date +%Y%m%d)",
    ]
    wheels    = true
    timeout_m = 60 * 6
  },
  {
    trigger_name     = "xla-nightly-38-tpu"
    image            = "xla"
    description      = "Build nightly image with TPU support"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version  = "3.8"
      arch            = "amd64"
      accelerator     = "tpu"
      package_version = "2.0"
    }
    image_tags = [
      "nightly_3.8_tpuvm",
      "nightly_3.8_$(date +%Y%m%d)",
    ]
    wheels    = true
    timeout_m = 60 * 6
  },
  {
    trigger_name = "xla-2-0-38-tpu"
    image        = "xla"
    description  = "Build v2.0.0 image with TPU support"
    # Don't use this tag here, since the repositiory at version v2.0.0
    # doesn't contain ansible setup. Instead, fetch PyTorch and XLAs sources at
    # the desired tag.
    # git_tag = "..."
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version = "3.8"
      arch           = "amd64"
      accelerator    = "tpu"
      # Fetch sources at the desired tag.
      pytorch_git_rev = "v2.0.0"
      xla_git_rev     = "r2.0"
      package_version = "2.0"
    }
    image_tags = [
      "r2.0_3.8_tpuvm",
    ]
    wheels    = true
    timeout_m = 60 * 6
  },
  {
    trigger_name = "xla-1-13-38-tpu"
    image        = "xla"
    description  = "Build v1.13.0 image with TPU support"
    # Don't use this tag here, since the repositiory at version v1.13.0
    # doesn't contain ansible setup. Instead, fetch PyTorch and XLAs sources at
    # the desired tag.
    # git_tag = "v1.13.0"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version = "3.8"
      arch           = "amd64"
      accelerator    = "tpu"
      # Fetch sources at the desired tag.
      pytorch_git_rev = "v1.13.0"
      xla_git_rev     = "v1.13.0"
      package_version = "1.13"
    }
    image_tags = [
      "r1.13_3.8_tpuvm",
    ]
    wheels    = true
    timeout_m = 60 * 6
  },
  {
    trigger_name = "xla-1-12-38-tpu"
    image        = "xla"
    description  = "Build v1.12.0 image with TPU support"
    # git_tag = "v1.12.0"
    trigger_schedule = "0 0 * * *"
    build_args = {
      python_version  = "3.8"
      arch            = "amd64"
      accelerator     = "tpu"
      pytorch_git_rev = "v1.12.0"
      xla_git_rev     = "v1.12.0"
      package_version = "1.12"
    }
    image_tags = [
      "r1.12_3.8_tpuvm",
    ]
    wheels    = true
    timeout_m = 60 * 6
  },
]

# Variables for the staging environment.

project_id = "tpu-pytorch"
public_docker_repo = {
  id = "docker-public-staging"
}
worker_pool = {
  name         = "worker-pool-staging"
  machine_type = "e2-standard-32"
}
storage_bucket_suffix          = "-staging"
build_runner_account_id_suffix = "-staging"
triggers_suffix                = "-staging"
