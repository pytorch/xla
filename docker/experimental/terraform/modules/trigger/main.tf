data "google_project" "project" { }

variable "release" {
  type = string
}

variable "branch" {
  type = string
  default = "master"
}

variable "python_version" {
  description = "Python version to use (e.g. 3.8)"
  type = string
}

variable "platform" {
  type = string
}

variable "docker_build_args" {
  type = list(string)
  default = [ "tpuvm=1" ]
}

variable "schedule" {
  type = string
  # Format: https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules
  default = "0 0 * * *"
}

variable "scheduler_service_account" {
  type = string
  default = null
}

variable "build_on_push" {
  type = string
  default = false
}

locals {
  trigger_name = format("pytorch-xla-%s-py%s-%s", replace(var.release, ".", "-"), replace(var.python_version, ".", ""), var.platform)
}

resource "google_cloudbuild_trigger" "build-trigger" {
  location = "global"
  name = local.trigger_name

  dynamic "github" {
    # HACK: `source_to_build` is mutually exclusive with `github`
    for_each = var.build_on_push ? [1] : []

    content {
      owner = "pytorch"
      name = "xla"
      push {
        # `branch` is treated as a regex, so look for exact match
        branch = "^${var.branch}$"
      }
    }
  }

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref = "refs/heads/${var.branch}"
  }

  git_file_source {
    path = "docker/experimental/cloudbuild.yaml"
    repo_type = "GITHUB"
    revision = "refs/heads/master"
    uri = "https://github.com/pytorch/xla"
  }

  substitutions = {
    _PLATFORM = var.platform
    _BUILD_ARGS = join(",", var.docker_build_args)
    _PYTHON_VERSION = var.python_version
  }
}

resource "google_cloud_scheduler_job" "trigger-schedule" {
  count = var.schedule != null ? 1 : 0

  name = format("%s-schedule", local.trigger_name)
  region = "us-central1"

  schedule = var.schedule
  time_zone = "America/Los_Angeles"

  http_target {
    http_method = "POST"
    uri = "https://cloudbuild.googleapis.com/v1/projects/${google_cloudbuild_trigger.build-trigger.project}/triggers/${google_cloudbuild_trigger.build-trigger.trigger_id}:run"

    oauth_token {
      service_account_email = var.scheduler_service_account
    }
  }
}
