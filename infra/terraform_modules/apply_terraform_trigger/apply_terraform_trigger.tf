module "cloud_build" {
  source = "../build_trigger"

  name                    = "provision-terraform"
  github_repo             = "pytorch/xla"
  scheduler_account_email = var.scheduler_account_email
  timeout_minutes         = var.timeout_minutes
  worker_pool_id          = var.worker_pool_id
  description             = var.description
  location                = var.location

  trigger_on_push     = var.trigger_on_push
  trigger_on_schedule = var.trigger_on_schedule

  steps = concat(
    local.fetch_ansible_build_config,
    local.build_and_push_docker_image_steps,
    length(var.wheels_srcs) > 0 ? local.collect_and_publish_wheels_steps : []
  )
}
