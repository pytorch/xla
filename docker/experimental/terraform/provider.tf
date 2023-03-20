# Run `gcloud auth application-default login` in your shell before
provider "google" {
  project = var.project_id
  region  = var.region
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.52.0"
    }
  }

  backend "gcs" {
    # TODO: This has to be changed to match current project or passed as cli
    # argument: "-backend-config="bucket=bucket_id"
    bucket = "tpu-pytorch-tfstate-staging"
    prefix = "terraform/state"
  }
}
