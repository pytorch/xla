variable "python_version" {
  type        = string
  default     = "3.8"
  description = "Python version for all docker images."
}

variable "public_docker_repo" {
  type = object({
    id       = string
    location = optional(string, "us-central1")
  })
}

variable "worker_pool" {
  type = object({
    name         = string
    location     = optional(string, "us-central1")
    machine_type = optional(string, "e2-standard-32")
    disk_size_gb = optional(number, 500)
  })
}

variable "storage_bucket_suffix" {
  type    = string
  default = ""
}

variable "build_runner_account_id_suffix" {
  type    = string
  default = ""
}

variable "project_id" {
  type        = string
  description = "ID of the GCP project."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "docker_images" {
  type = list(
    object({
      trigger_name = optional(string, "")

      # Name of the produced docker image (without tags).
      image = string

      # Branch to fetch the Ansible setup at (not the XLA source code!)
      branch = optional(string, "master")

      # Dockerfile path withing docker context (`dir` parameter).
      dockerfile = optional(string, "Dockerfile")

      # Git tag to fetch the Ansible setup at (not the XLA source code!)
      # Contents of the `dir` directory will be fetched at this tag.
      git_tag = optional(string, "")

      # Cloud Build trigger description (for human consumption).
      description = optional(string, "")

      # Trigger build only if any of the following was modified in the specified
      # `branch` or `tag`.
      trigger_files = optional(list(string), [])

      # Trigger build on the specified cron schedule.
      trigger_schedule = optional(string, "")

      # Base directory for docker context.
      dir = optional(string, "docker/experimental/ansible")

      # Build args to pass to the dockerfile (`ARG build_arg=`).
      build_args = optional(map(any), {})

      # Tags for the produced docker image.
      # Can include bash expression e.g. "my_tag_$(date +%Y%m%d)".
      image_tags = optional(list(string), [])

      # Set to true, if any *.whl files from /dist should be uploaded to
      # the public storage bucket.
      wheels = optional(bool, false)

      # Build job timeout.
      timeout_m = optional(number, 30)
    })
  )

  validation {
    condition = alltrue([
      for di in var.docker_images : (di.branch == "") != (di.git_tag == "")
    ])
    error_message = "Specify exactly one of `branch` or `git_tag` for each docker image."
  }
}

variable "triggers_suffix" {
  type    = string
  default = ""
}
