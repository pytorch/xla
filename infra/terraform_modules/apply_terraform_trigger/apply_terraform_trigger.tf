variable "config_directory" {
  type = string
}

variable "include_files" {
  type    = list(string)
  default = []
}

variable "branch" {
  type = string
}

variable "description" {
  default = ""
}

variable "worker_pool_id" {
  type = string
}

variable "name" {
  type = string
}

module "cloud_build" {
  source = "../build_trigger"

  name            = var.name
  github_repo     = "pytorch/xla"
  timeout_minutes = 60
  worker_pool_id  = var.worker_pool_id
  description     = var.description

  trigger_on_push = { branch = var.branch }

  steps = [
    {
      id         = "apply_terraform"
      name       = "hashicorp/terraform:latest"
      entrypoint = "sh"
      args = [
        "-c", join(" ", [
          "terraform -chdir=${var.config_directory} init &&",
          "terraform -chdir=${var.config_directory} apply -auto-approve",
        ])
      ]
    },
  ]
}
