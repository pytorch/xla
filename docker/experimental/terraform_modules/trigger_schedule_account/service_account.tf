resource "google_service_account" "build_runner" {
  account_id  = "build_triggers_scheduler"
  description = "Service account for Scheduled Jobs. Has permissions to trigger Cloud Builds."
}

resource "google_project_iam_custom_role" "build_runner" {
  role_id     = "build_runner"
  title       = "Build Runner"
  description = "Grants permissions to trigger Cloud Builds."
  permissions = ["cloudbuild.builds.create"]
}

resource "google_project_iam_member" "build_runner" {
  role   = google_project_iam_custom_role.build_runner.name
  member = "serviceAccount:${google_service_account.build_runner.email}"
}
