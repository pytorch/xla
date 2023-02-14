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

variable "storage_bucket_suffix" {
  type = string
  default = ""
}

variable "build_runner_account_id_suffix" {
  type = string
  default = ""
}

variable "worker_pool" {
  # TODO: Variablize this.
  default = "projects/tpu-pytorch/locations/us-central1/workerPools/wheel_build"
}

variable "project_id" {
  type = string
  description = "ID of the GCP project."
}