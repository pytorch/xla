# Storage bucket for Terraform state of this project.
module "tfstate_storage_bucket" {
  source = "../terraform_modules/storage_bucket"
  name   = "tpu-ci-tfstate"
}

# Private worker pool for Cloud Builds.
module "worker_pool" {
  source = "../terraform_modules/worker_pool"
  # See https://cloud.google.com/compute/docs/machine-resource#machine_type_comparison.
  machine_type = "n1-standard-4"
}
