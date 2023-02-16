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

variable "region" {
  type = string
  default = "us-central1"
}

variable "docker_images" {
  type = list(
    object({
      image = string
      branch = optional(string, "")
      dockerfile = optional(string, "Dockerfile")
      git_tag = optional(string, "")
      description = optional(string, "")
      trigger_files = optional(list(string), [])
      trigger_schedule = optional(string, "")
      dir = optional(string, "")
      build_args=optional(map(any), {})
      image_tags = optional(list(string), [])
      timeout_m = optional(number, 30)
    })
  )

  # validation {
  #   condition     = anytrue([
  #     for di in var.docker_images: (di.branch == "" && di.git_tag == "") || (di.branch != "" && di.git_tag != "")
  #     ])
  #   error_message = "Specify exactly one of `branch` or `git_tag` for each docker image."
  # }
}

variable "triggers_suffix" {
  type = string
  default = ""
}