provider "google" {
  project = "tpu-pytorch"
  region  = "us-central-1"
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.6.0"
    }
  }

  required_version = ">= 0.14"
  backend "gcs" {
    bucket = "tpu-ci-tfstate"
    prefix = "terraform/state"
  }
}
