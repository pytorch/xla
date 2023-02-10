variable "python_version" {
  type = string
  default = "3.8"
  description = "Python version for all docker images."
}

variable "image_repository" {
  # TODO: Variablize this.
  default = "us-central2-docker.pkg.dev/core-ml-engprod-build-farm/docker-repo"
}

variable "project_id" {
  type = string
  description = "ID of the GCP project."
}