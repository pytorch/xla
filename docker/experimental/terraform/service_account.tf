resource "google_service_account" "build_runner" {
  project    = var.project_id
  account_id = "build-scheduler${var.build_runner_account_id_suffix}"
}

resource "google_project_iam_member" "build_runner_build_editor" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.build_runner.email}"
}

resource "google_project_iam_custom_role" "build_runner" {
  project     = var.project_id
  role_id     = "buildRunner"
  title       = "Build Runner"
  description = "Grants permissions to trigger Cloud Builds."
  permissions = ["cloudbuild.builds.create"]
}
