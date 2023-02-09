resource "google_storage_bucket" "tfstate" {
  name          = "${var.project_id}-tfstate"
  force_destroy = false
  location      = "US"
  storage_class = "STANDARD"

  # Required by project policy.
  # See https://cloud.google.com/storage/docs/uniform-bucket-level-access.
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}