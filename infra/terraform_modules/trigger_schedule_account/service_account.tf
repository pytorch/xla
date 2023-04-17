resource "google_service_account" "build_runner" {
  account_id  = "build-triggers-scheduler"
  description = "Service account for Scheduled Jobs. Has permissions to trigger Cloud Builds."
}

resource "google_project_iam_custom_role" "build_runner" {
  role_id     = "build_runner"
  title       = "Build Runner"
  description = "Grants permissions to trigger Cloud Builds."
  permissions = ["cloudbuild.builds.create"]
}

data "google_project" "project" {}

resource "google_project_iam_member" "build_runner" {
  role    = google_project_iam_custom_role.build_runner.name
  project = data.google_project.project.project_id
  member  = "serviceAccount:${google_service_account.build_runner.email}"
}

output "email" {
  value = google_service_account.build_runner.email
}
