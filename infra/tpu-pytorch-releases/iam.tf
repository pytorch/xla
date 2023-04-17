data "google_project" "project" {}

variable "project_admins" {
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
