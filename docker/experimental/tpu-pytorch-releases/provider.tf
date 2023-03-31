# Run `gcloud auth application-default login` before running Terraform.
provider "google" {
  project = "tpu-pytorch-releases"
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
    # Make sure that bucket name matches the only specified in ./storage_buckets.tf.
    bucket = "tpu-pytorch-releases-tfstate"
    prefix = "terraform/state"
  }
}
