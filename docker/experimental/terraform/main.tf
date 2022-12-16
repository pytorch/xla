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

resource "google_cloudbuild_trigger" "tpu-test-trigger" {
  location = "global"
  name = "ci-tpu-test-trigger"

  github {
    owner = "pytorch"
    name  = "xla"
    push {
      branch = "^master$"
    }
  }

  filename = "test/tpu/cloudbuild.yaml"
}

module "nightly-py37-tpuvm" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.7"
  platform = "tpuvm"
  docker_build_args = [ "tpuvm=1" ]
}

module "nightly-py38-tpuvm" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "tpuvm"
  docker_build_args = [ "tpuvm=1" ]
}

module "nightly-py38-tpunode" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "tpunode"
  docker_build_args = [ "tpuvm=0" ]
}

module "nightly-py38-cuda112" {
  source = "./modules/trigger"

  release = "nightly"
  python_version = "3.8"
  platform = "cuda112"
  docker_build_args = [ "tpuvm=0,cuda=1"]
}
