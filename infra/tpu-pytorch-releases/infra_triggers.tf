module "terraform_apply" {
  source = "../terraform_modules/apply_terraform_trigger"

  include_files    = ["infra/**"]
  name             = "terraform-provision-trigger"
  branch           = "master"
  description      = "Trigger that provisions Terraform setup in infra/tpu-pytorch-releases/cloud_builds.tf"
  config_directory = "infra/tpu-pytorch-releases"

  worker_pool_id = module.worker_pool.id
}
