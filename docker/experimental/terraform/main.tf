provider "google" {
  project = "tpu-pytorch"
}

resource "random_id" "bucket_prefix" {
  byte_length = 8
}

resource "google_storage_bucket" "default" {
  name          = "${random_id.bucket_prefix.hex}-bucket-tfstate"
  force_destroy = false
  location      = "US"
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
}

terraform {
 backend "gcs" {
   bucket  = "426a09baf5992b6a-bucket-tfstate"
   prefix  = "terraform/state"
 }
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

resource "google_cloudbuild_worker_pool" "gcb-pool" {
  name        = "wheel_build"
  location    = "us-central1"

  worker_config {
    disk_size_gb   = 500
    machine_type   = "e2-standard-32"
    no_external_ip = false
  }
}

resource "google_service_account" "cloud-build-trigger-scheduler" {
  account_id   = "cloud-build-trigger-scheduler"
  display_name = "Cloud Build Trigger Scheduler"
  description  = "Service account for running Cloud Build triggers in a Cloud Scheduler job"
}

resource "google_project_iam_member" "cloud-build-scheduler-permission" {
  project = google_service_account.cloud-build-trigger-scheduler.project
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.cloud-build-trigger-scheduler.email}"
}

module "nightly-py37-tpuvm" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.7"
  platform = "tpuvm"
  docker_build_args = [ "tpuvm=1" ]
  scheduler_service_account = google_service_account.cloud-build-trigger-scheduler.email
}

module "nightly-py38-tpuvm" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "tpuvm"
  docker_build_args = [ "tpuvm=1" ]
  scheduler_service_account = google_service_account.cloud-build-trigger-scheduler.email
}

module "nightly-py38-tpunode" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "tpunode"
  docker_build_args = [ "tpuvm=0" ]
  scheduler_service_account = google_service_account.cloud-build-trigger-scheduler.email
}

module "nightly-py38-cuda112" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "cuda112"
  docker_build_args = [ "tpuvm=0,cuda=1"]
  scheduler_service_account = google_service_account.cloud-build-trigger-scheduler.email
}

module "r113-py37-tpuvm" {
  source = "./modules/trigger"

  release = "1.13"
  branch = "wcromar/r1.13-kaggle"
  build_on_push = true
  schedule = null
  python_version = "3.7"
  platform = "tpuvm"
  docker_build_args = [ "tpuvm=1" ]
  scheduler_service_account = google_service_account.cloud-build-trigger-scheduler.email
}
