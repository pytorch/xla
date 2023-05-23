module "terraform_apply" {
  source = "../terraform_modules/apply_terraform_trigger"

  included_files    = ["infra/**"]
  branch           = "master"
  config_directory = "infra/tpu-pytorch"

  worker_pool_id = module.worker_pool.id
  location       = "global"
}
