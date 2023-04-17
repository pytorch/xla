variable "name" {
  default = "main"
}

variable "location" {
  default = "us-central1"
}

variable "machine_type" {
  default = "e2-standard-32"
}

variable "disk_size_gb" {
  default = 500
}

resource "google_cloudbuild_worker_pool" "worker_pool" {
  name     = var.name
  location = var.location

  worker_config {
    disk_size_gb   = var.disk_size_gb
    machine_type   = var.machine_type
    no_external_ip = false
  }
}

output "id" {
  value = google_cloudbuild_worker_pool.worker_pool.id
}
