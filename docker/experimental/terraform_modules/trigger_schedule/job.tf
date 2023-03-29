# Provides a scheduled job that will trigger Cloud Build periodically.

variable "schedule" {
  description = "Job schedule in cron format https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules"
  type        = string
  default     = "0 0 * * *"
}

variable "trigger" {
  description = "An instance of google_cloudbuild_trigger for which the schedule job should be configured."
  type = object({
    project    = string
    trigger_id = string
    name       = string
  })
}

variable "time_zone" {
  description = "The schedule will be relative to this time zone."
  default     = "America/Los_Angeles"
  type        = string
}

variable "scheduler_service_account" {
  type = string
}

resource "google_service_account" "build_runner" {
  account_id = "build_runner"
}

resource "google_project_iam_custom_role" "build_runner" {
  role_id     = "build_runner"
  title       = "Build Runner"
  description = "Grants permissions to trigger Cloud Builds."
  permissions = ["cloudbuild.builds.create"]
}

resource "google_project_iam_member" "build_runner" {
  role    = google_project_iam_custom_role.build_runner.name
  member  = "serviceAccount:${google_service_account.build_runner.email}"
  condition {
    expression =
  }
}

resource "google_cloud_scheduler_job" "trigger-schedule" {
  name      = format("%s-schedule", var.trigger.name)
  schedule  = var.schedule
  time_zone = "America/Los_Angeles"

  http_target {
    http_method = "POST"
    uri         = "https://cloudbuild.googleapis.com/v1/projects/${var.trigger.project}/triggers/${var.trigger.trigger_id}:run"

    oauth_token {
      service_account_email = var.scheduler_service_account
    }
  }
}
