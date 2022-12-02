variable "release" {
  type = string
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

locals {
  trigger_name = format("pytorch-xla-%s-py%s-%s", var.release, replace(var.python_version, ".", ""), var.platform)
}

resource "google_cloudbuild_trigger" "build-trigger" {
  location = "global"
  name = local.trigger_name

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    # TODO: make branch configurable
    ref = "refs/heads/master"
  }

  git_file_source {
    path = "docker/experimental/cloudbuild.yaml"
    repo_type = "GITHUB"
    # TODO: make branch configurable
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
  name = format("%s-schedule", local.trigger_name)
  region = "us-central1"

  schedule = "0 0 * * *"
  time_zone = "America/Los_Angeles"

  http_target {
    http_method = "POST"
    uri = "https://cloudbuild.googleapis.com/v1/projects/${google_cloudbuild_trigger.build-trigger.project}/triggers/${google_cloudbuild_trigger.build-trigger.trigger_id}:run"

    oauth_token {
      # TODO: Use a better service account
      service_account_email = "164006649440-compute@developer.gserviceaccount.com"
    }
  }
}
