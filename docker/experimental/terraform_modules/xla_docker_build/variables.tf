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
  default = []
  type    = list(string)
}

variable "sources_git_rev" {
  default = "master"
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

variable "schedule" {
  default = ""
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
