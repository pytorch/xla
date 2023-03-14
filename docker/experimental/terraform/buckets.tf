resource "google_storage_bucket" "tfstate" {
  name          = "${var.project_id}-tfstate${var.storage_bucket_suffix}"
  force_destroy = false
  location      = "US"
  storage_class = "STANDARD"

  # Required by project policy.
  # See https://cloud.google.com/storage/docs/uniform-bucket-level-access.
  uniform_bucket_level_access = false

  versioning {
    enabled = true
  }
}

# Storage bucket for all publicly released wheels.
resource "google_storage_bucket" "public_wheels" {
  name          = "${var.project_id}-wheels-public"
  force_destroy = false
  location      = "US"
  storage_class = "STANDARD"

  uniform_bucket_level_access = false

  versioning {
    enabled = true
  }
}

# Grants all users (public) read access to the bucket with wheels.
resource "google_storage_bucket_access_control" "all_users_read_public_wheels" {
  bucket = google_storage_bucket.public_wheels.name
  role   = "READER"
  entity = "allUsers"
}

output "public_wheels_bucket_url" {
  value = google_storage_bucket.public_wheels.url
}

output "tfstate_bucket_url" {
  value = google_storage_bucket.tfstate.url
}