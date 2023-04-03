# Docker registry for official images.
module "docker_registry" {
  source      = "../terraform_modules/docker_registry"
  name        = "docker"
  description = "Private docker images for PyTorch/XLA."
  # public_read_access = true
}

# Storage bucket for Terraform state of this project.
module "tfstate_storage_bucket" {
  source = "../terraform_modules/storage_bucket"
  name   = "tpu-pytorch-tfstate"
}

# Private worker pool for Cloud Builds.
module "worker_pool" {
  source = "../terraform_modules/worker_pool"
  # See https://cloud.google.com/compute/docs/machine-resource#machine_type_comparison.
  machine_type = "e2-standard-32"
}
