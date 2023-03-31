# Docker registry for official images.
module "docker_registry" {
  source             = "../terraform_modules/docker_registry"
  name               = "docker"
  description        = "Official docker images for PyTorch/XLA."
  public_read_access = true
}

# Public storage bucket for PyTorch/XLA wheels.
module "releases_storage_bucket" {
  source             = "../terraform_modules/storage_bucket"
  name               = "pytorch-xla-releases"
  # public_read_access = true
}

# Storage bucket for Terraform state of this project.
module "tfstate_storage_bucket" {
  source = "../terraform_modules/storage_bucket"
  name   = "tpu-pytorch-releases-tfstate"
}

# Private worker pool for Cloud Builds.
module "worker_pool" {
  source = "../terraform_modules/worker_pool"
  # See https://cloud.google.com/compute/docs/machine-resource#machine_type_comparison.
  machine_type = "e2-standard-32"
}

# Service account for Scheduler Job that triggers Cloud Builds.
module "scheduler_account" {
  source = "../terraform_modules/trigger_schedule_account"
}
