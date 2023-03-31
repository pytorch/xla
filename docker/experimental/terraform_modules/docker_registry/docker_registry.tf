# Docker repository in Artifact Registry.

variable "name" {
  type = string
}

variable "description" {
  default = ""
}

variable "public_read_access" {
  default     = false
  description = "Set to true to allow public read access to the docker repository."
}

variable "location" {
  default = "us-central1"
}

resource "google_artifact_registry_repository" "docker" {
  repository_id = var.name
  location      = var.location
  description   = var.description
  format        = "DOCKER"
}

resource "google_artifact_registry_repository_iam_member" "public_read_access" {
  count = var.public_read_access ? 1 : 0

  role       = "roles/artifactregistry.reader"
  member     = "allUsers"
  repository = google_artifact_registry_repository.docker.name
}

locals {
  repo = google_artifact_registry_repository.docker
}

output "url" {
  value = "${local.repo.location}-docker.pkg.dev/${local.repo.project}/${local.repo.repository_id}"
}
