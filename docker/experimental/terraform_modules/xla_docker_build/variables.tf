variable "trigger_name" {
  type = string
}

variable "image_name" {
  type = string
}

variable "description" {
  type    = string
  default = ""
}

variable "image_tags" {
  type = list(string)
}

variable "location" {
  default = "us-central1"
}

variable "description" {
  type    = string
  default = ""
}

variable "ansible_branch" {
  default = "master"
}

variable "include_files" {
  default = null
  type    = list(string)
}

variable "build_args" {
  type        = map(any)
  description = "Build args to pass to the dockerfile (`ARG build_arg=`)."
  default     = {}
}

variable "ansible_vars" {
  type        = map(any)
  description = "Variables passed to Ansible playbook. Encoded as JSON."
}

variable "trigger_on_push" {
  type = object({
    branch         = optional(string)
    tag            = optional(string)
    included_files = optional(list(string), [])
  })
  default = null
}

variable "trigger_on_schedule" {
  type = object({
    schedule = string
    branch   = optional(string)
    tag      = optional(string)
  })
  default = null
}

variable "scheduler_account_email" {
  default = ""
}

variable "timeout_minutes" {
  default = 360
}

variable "dockerfile" {
  default = "Dockerfile"
}

variable "docker_context_dir" {
  default = "docker/experimental/ansible"
}

variable "docker_repo_url" {
  type = string
}

variable "wheels_srcs" {
  default = []
  type    = list(string)
}

variable "wheels_dest" {
  default = ""
}

variable "worker_pool_id" {
  type = string
}
