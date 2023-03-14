resource "google_cloudbuild_worker_pool" "worker-pool" {
  name        = var.worker_pool.name
  location    = var.worker_pool.location

  worker_config {
    disk_size_gb   = var.worker_pool.disk_size_gb
    machine_type   = var.worker_pool.machine_type
    no_external_ip = false
  }
}

locals {
  worker_pool_id = google_cloudbuild_worker_pool.worker-pool.id
}

output "worker_pool_id" {
  value = local.worker_pool_id
}