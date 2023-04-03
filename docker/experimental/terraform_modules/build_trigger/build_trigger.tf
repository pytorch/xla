variable "github_repo" {
  type        = string
  description = <<EOT
    GitHub repository name, without the github.com prefix.
    Expected format: "owner/repo" (should include exactly one slash)
    Examples:
    * for https://github.com/pytorch/xla set "pytorch/xla"
    * for https://github.com/pytorch/pytorch set "pytorch/pytorch"
  EOT
}

variable "name" {
  type = string
}

variable "description" {
  default = ""
}

variable "worker_pool_id" {
  type = string
}

variable "trigger_on_push" {
  type = object({
    branch         = optional(string)
    tag            = optional(string)
    included_files = optional(list(string), [])
  })
  default = null
}

variable "trigger_on_schedule" {
  type = object({
    schedule = string
    branch   = optional(string)
    tag      = optional(string)
  })
  default = null
}

variable "steps" {
  type = list(object({
    id         = optional(string)
    entrypoint = optional(string)
    name       = string
    dir        = optional(string)
    args       = optional(list(string))
    volumes = optional(list(
      object({
        name = string
        path = string
    })), [])
  }))
}

variable "scheduler_account_email" {
  default = ""
}

variable "timeout_minutes" {
  default = 60
}

variable "location" {
  default = "us-central1"
}

locals {
  github_repo_parts = split("/", var.github_repo)
}

# Detailed documentation on CloudBuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build
resource "google_cloudbuild_trigger" "trigger" {
  name        = var.name
  description = var.description
  location    = var.location

  # Source (context) in which build trigger will run when triggered on branch or tag push.
  # The source exact version will be the commit which caused the trigger event (push).
  dynamic "github" {
    # Include this block only if trigger on push is configured.
    for_each = var.trigger_on_push != null ? [1] : []

    content {
      owner = local.github_repo_parts[0]
      name  = local.github_repo_parts[1]

      push {
        # The `branch` and `tag` fields are regexes, so look for exact match.
        branch = var.trigger_on_push.branch != null ? "^${var.trigger_on_push.branch}$" : null
        tag    = var.trigger_on_push.tag != null ? "^${var.trigger_on_push.tag}$" : null
      }
    }
  }

  # Source (context) in which build trigger will run when triggered on schedule.
  dynamic "source_to_build" {
    # Include this block only if trigger on schedule is configured.
    for_each = var.trigger_on_schedule != null ? [1] : []

    content {
        uri       = "https://github.com/${var.github_repo}"
        repo_type = "GITHUB"
        ref = (
          var.trigger_on_schedule.branch != ""
          ? "refs/heads/${var.trigger_on_schedule.branch}"
          : "refs/tags/${var.trigger_on_schedule.tag}"
        )
    }
  }

  included_files = var.trigger_on_push != null ? var.trigger_on_push.included_files : null

  build {
    dynamic "step" {
      for_each = var.steps

      content {
        id         = step.value.id
        entrypoint = step.value.entrypoint
        name       = step.value.name
        args       = step.value.args
        dir        = step.value.dir

        dynamic "volumes" {
          iterator = volume
          for_each = step.value.volumes

          content {
            name = volume.value.name
            path = volume.value.path
          }
        }
      }
    }

    options {
      substitution_option   = "ALLOW_LOOSE"
      dynamic_substitutions = true
      worker_pool           = var.worker_pool_id
    }

    timeout = "${var.timeout_minutes * 60}s"
  }

  include_build_logs = var.trigger_on_push != null ? "INCLUDE_BUILD_LOGS_WITH_STATUS" : null
}

# Add scheduled job if the build should be triggered on schedule
module "schedule_triggers" {
  source = "../trigger_schedule_job"
  count  = var.trigger_on_schedule != null ? 1 : 0

  trigger                 = google_cloudbuild_trigger.trigger
  schedule                = var.trigger_on_schedule.schedule
  scheduler_account_email = var.scheduler_account_email
}
