variable "cluster_name" {
  description = "Name of the Container Cluster containing the v4 node pool"
  type = string
}

variable "cpu_nodepool_name" {
  description = "Name of the CPU Nodepool"
  type = string
}

variable "cpu_node_count" {
  description = "Number of CPU nodes"
  type = number
}

variable "tpu_nodepool_name" {
  description = "Name of the TPU Nodepool"
  type = string
}

variable "max_tpu_nodes" {
  description = "Maximum number of TPU nodes and runners"
  type = number
}

variable "arc_namespace" {
  description = "The namespace where ARC will reside"
  default = "arc-systems"
  type = string
}

variable "runner_namespace" {
  description = "The namespace where the ARC runners will reside"
  default = "arc-runners"
  type = string
}

variable "github_repo_url" {
  description = "The full URL of the repository which will be utilizing the self-hosted runners in ARC"
  type = string
}

variable "project_id" {
  description = "The project ID"
  type = string
}

variable "runner_image" {
  description = "The Docker image used in the self-hosted runner"
  type = string
}
