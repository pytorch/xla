variable "python_version" {
  type = string
  default = "3.8"
  description = "Python version for all docker images."
}

variable "public_docker_repo" {
  type = object({
    id = string
    location = optional(string, "us-central1")
  })
}

variable "worker_pool" {
  type = object({
    name = string
    location = optional(string, "us-central1")
    machine_type = optional(string, "e2-standard-32")
    disk_size_gb = optional(number, 500)
  })
}

variable "storage_bucket_suffix" {
  type = string
  default = ""
}

variable "build_runner_account_id_suffix" {
  type = string
  default = ""
}

variable "project_id" {
  type = string
  description = "ID of the GCP project."
}