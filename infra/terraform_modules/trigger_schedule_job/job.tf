# Provides a scheduled job that will trigger Cloud Build periodically.

variable "schedule" {
  description = "Job schedule in cron format https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules"
  type        = string
  default     = "0 0 * * *"
}

variable "trigger" {
  description = "An instance of google_cloudbuild_trigger for which the schedule job should be configured."
  type = object({
    id   = string
    name = string
  })
}

variable "time_zone" {
  description = "The schedule will be relative to this time zone."
  default     = "America/Los_Angeles"
  type        = string
}

variable "scheduler_account_email" {
  type = string
}

resource "google_cloud_scheduler_job" "trigger_schedule" {
  name      = format("%s-schedule", var.trigger.name)
  schedule  = var.schedule
  time_zone = var.time_zone

  http_target {
    http_method = "POST"
    uri         = "https://cloudbuild.googleapis.com/v1/${var.trigger.id}:run"

    oauth_token {
      service_account_email = var.scheduler_account_email
    }
  }
}
