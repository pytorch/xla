variable "name" {
  type = string
}

variable "public_read_access" {
  default     = false
  description = "Set to true to allow public read access to the bucket."
}

variable "location" {
  default = "US"
}

variable "storage_class" {
  default = "STANDARD"
}

resource "google_storage_bucket" "bucket" {
  name = var.name
  # If you try to delete a bucket that contains objects, Terraform will fail
  # that run. Delete all bucket objects before deleting the bucket:
  # gsutil -m rm -r gs://my-bucket/*
  force_destroy = false
  location      = var.location
  storage_class = var.storage_class

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

data "google_iam_policy" "public_read_access" {
  count = var.public_read_access ? 1 : 0

  binding {
    role = "roles/storage.objectViewer"
    members = [
      "allUsers",
    ]
  }
}

resource "google_storage_bucket_iam_policy" "policy" {
  count = var.public_read_access ? 1 : 0

  bucket      = google_storage_bucket.bucket.name
  policy_data = data.google_iam_policy.public_read_access.policy_data
}

output "url" {
  value = google_storage_bucket.bucket.url
}
