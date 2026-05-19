data "google_project" "project" {}

variable "project_admins" {
  type    = list(string)
  default = []
}

variable "cloudbuild_editors" {
  type    = list(string)
  default = []
}

variable "project_remote_build_writers" {
  type        = list(string)
  description = "List of members with remote build writer access"
  default     = []
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

resource "google_project_iam_member" "project_remote_build_writers" {
  for_each = toset(var.project_remote_build_writers)
  project  = "tpu-pytorch-releases"
  role     = "projects/tpu-pytorch-releases/roles/remoteBuildWriterRole"
  member   = each.value
}

resource "google_project_iam_custom_role" "remote_build_writer_role" {
  role_id     = "remoteBuildWriterRole"
  title       = "Remote Build Writer"
  description = "Custom role for remote build writers"
  permissions = [
    "remotebuildexecution.actions.create",
    "remotebuildexecution.actions.get",
    "remotebuildexecution.actions.set",
    "remotebuildexecution.blobs.create",
    "remotebuildexecution.blobs.get",
    "remotebuildexecution.logstreams.create",
    "remotebuildexecution.logstreams.get",
    "remotebuildexecution.logstreams.update"
  ]
}