# Run `gcloud auth application-default login` before running Terraform.
provider "google" {
  project = "tpu-pytorch"
  region  = "us-central1"
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.52.0"
    }
  }

  backend "gcs" {
    # Make sure that bucket name matches the only specified in ./misc.tf.
    bucket = "tpu-pytorch-tfstate"
    prefix = "terraform/state"
  }
}
