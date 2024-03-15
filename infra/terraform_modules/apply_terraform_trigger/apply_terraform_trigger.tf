variable "config_directory" {
  type = string
}

variable "included_files" {
  type    = list(string)
  default = []
}

variable "branch" {
  type = string
}

variable "worker_pool_id" {
  type = string
}

variable "name" {
  type    = string
  default = "terraform-provision-trigger"
}

data "google_project" "project" {}

resource "google_service_account" "runner_account" {
  account_id   = "terraform-applier"
  display_name = "Runner of ${var.name}"
  description  = "Service Account that runs ${var.name} trigger. Has roles/editor permissions"
}

resource "google_project_iam_member" "runner_is_editor" {
  for_each = toset([
    "roles/storage.admin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/editor",
  ])

  project = data.google_project.project.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runner_account.email}"
}

variable "location" {
  default = "us-central1"
}

module "cloud_build" {
  source = "../build_trigger"

  name            = var.name
  github_repo     = "pytorch/xla"
  timeout_minutes = 60
  worker_pool_id  = var.worker_pool_id
  description     = "Trigger that provisions Terraform setup in ${var.config_directory}."
  service_account = google_service_account.runner_account.id
  logging         = "CLOUD_LOGGING_ONLY"
  trigger_on_push = {
    branch         = var.branch
    included_files = var.included_files
  }
  location = var.location

  steps = [
    {
      id         = "apply_terraform"
      name       = "hashicorp/terraform:latest"
      entrypoint = "sh"
      args = [
        "-c", join(" ", [
          "terraform -chdir=${var.config_directory} init -no-color &&",
          "terraform -chdir=${var.config_directory} apply -no-color -auto-approve",
        ])
      ]
    },
  ]
}
