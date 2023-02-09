# Terraform for CloudBuild triggers

This Terraform sets up:
- storage bucket for PyTorch and PyTorch/XLA wheels.
- storage bucket for Terraform state.
- TODO: sth for docker images?

# Prerequisites

1. Run `gcloud auth application-default login` on your local workstation.
2. Make sure Terraform binary is installed.