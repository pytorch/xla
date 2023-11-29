# Storage bucket for Terraform state of this project.
module "tfstate_storage_bucket" {
  source = "../terraform_modules/storage_bucket"
  name   = "tpu-ci-tfstate"
}
