data "google_project" "project" {}

variable "project_admins" {
  type    = list(string)
  default = []
}

variable "cloudbuild_editors" {
  type    = list(string)
  default = []
}

resource "google_project_iam_member" "project_iam_admins" {
  for_each = toset(var.project_admins)

  project = data.google_project.project.project_id
  role    = "roles/resourcemanager.projectIamAdmin"
  member  = each.value
}

resource "google_project_iam_member" "storage_admins" {
  for_each = toset(var.project_admins)

  project = data.google_project.project.project_id
  role    = "roles/storage.admin"
  member  = each.value
}

resource "google_project_iam_member" "cloudbuild_editor" {
  for_each = toset(var.cloudbuild_editors)

  project = data.google_project.project.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = each.value
}

resource "google_project_iam_custom_role" "remote_bazel_role" {
  role_id     = "remoteBuildWriterRole"
  title       = "Remote Build Writer"
  description = "For running remote bazel builds and read/write from remote cache on GCP."
  stage       = "ALPHA"
  permissions = [
    "remotebuildexecution.actions.create",
    "remotebuildexecution.actions.get",
    "remotebuildexecution.actions.set",
    "remotebuildexecution.blobs.create",
    "remotebuildexecution.blobs.get",
    "remotebuildexecution.logstreams.create",
    "remotebuildexecution.logstreams.get",
    "remotebuildexecution.logstreams.update",
  ]
}

data "google_project" "project" {}

variable "project_remote_build_writers" {
  type    = list(string)
  default = []
}

resource "google_project_iam_member" "project_remote_build_writers" {
  for_each = toset(var.project_remote_build_writers)

  project = data.google_project.project.project_id
  role    = google_project_iam_custom_role.remote_bazel_role.id
  member  = each.value
}
