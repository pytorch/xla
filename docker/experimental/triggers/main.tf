provider "google" {
  project = "tpu-pytorch"
}

resource "google_artifact_registry_repository" "torch-xla-python-repo" {
  location      = "us"
  repository_id = "torch-xla"
  description   = "PyTorch/XLA nightly packages"
  format        = "PYTHON"
}

resource "google_artifact_registry_repository" "torch-xla-docker-repo" {
  location      = "us"
  repository_id = "torch-xla-images"
  description   = "PyTorch/XLA nightly images"
  format        = "DOCKER"
}

resource "google_cloudbuild_trigger" "tpu-vm-build" {
  location = "global"
  name = "wcromar-test-trigger"

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref = "refs/heads/master"
  }

  git_file_source {
    path = "docker/experimental/cloudbuild.yaml"
    repo_type = "GITHUB"
    revision = "refs/heads/master"
    uri = "https://github.com/pytorch/xla"
  }

  substitutions = {
    _BUILD_ARGS = "tpuvm=1,cuda=0"
  }
}

resource "google_cloud_scheduler_job" "wcromar-nightly" {
  name = "wcromar-test-trigger-schedule"
  region = "us-central1"

  schedule = "0 0 * * *"
  time_zone = "America/Los_Angeles"

  http_target {
    http_method = "POST"
    uri = "https://cloudbuild.googleapis.com/v1/projects/${google_cloudbuild_trigger.tpu-vm-build.project}/triggers/${google_cloudbuild_trigger.tpu-vm-build.trigger_id}:run"

    oauth_token {
      service_account_email = "164006649440-compute@developer.gserviceaccount.com"
    }
  }
}
