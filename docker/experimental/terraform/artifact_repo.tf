# Docker repository in Artifact Registry for all public images.
resource "google_artifact_registry_repository" "public_docker_repo" {
  location      = var.public_docker_repo.location
  repository_id = var.public_docker_repo.id
  description   = "Official docker images."
  format        = "DOCKER"
}

resource "google_artifact_registry_repository_iam_member" "all_users_read_public_docker_repo" {
  role       = "roles/artifactregistry.reader"
  member     = "allUsers"
  project    = google_artifact_registry_repository.public_docker_repo.project
  location   = google_artifact_registry_repository.public_docker_repo.location
  repository = google_artifact_registry_repository.public_docker_repo.name
}

locals {
  public_repo            = google_artifact_registry_repository.public_docker_repo
  public_docker_repo_url = "${local.public_repo.location}-docker.pkg.dev/${var.project_id}/${local.public_repo.repository_id}"
}

output "public_docker_registry_url" {
  value = local.public_docker_repo_url
}
