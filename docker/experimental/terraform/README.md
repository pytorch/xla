# Terraform for CloudBuild triggers

This Terraform setup provisions:
- public storage bucket for PyTorch and PyTorch/XLA wheels.
- private storage bucket for Terraform state.
- public artifact repository for docker images.
- cloud builds for nightly and release docker images and wheels.
- schedule jobs and a service account for triggering cloud build.

# Running

1. Run `gcloud auth application-default login` on your local workstation.
2. Make sure that a recent Terraform binary is installed (>= 1.3.8).
   If not, install Terraform from the [official source](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).
3. Run `terraform apply -var-file=vars/staging.tfvars`.


