variable "sources_git_rev" {
  default = "master"
}

variable "ansible_git_rev" {
  default = "master"
}

variable "image_name" {
  type = string
}

variable "image_tags" {
  default = []
  type    = list(string)
}

variable "arch" {
  default = ""
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
