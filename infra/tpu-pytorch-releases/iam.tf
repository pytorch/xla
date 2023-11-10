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
