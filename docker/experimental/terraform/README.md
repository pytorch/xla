# Terraform configuration for build/test resources

Download the latest Terraform binary for your system and add it to your `$PATH`:
https://developer.hashicorp.com/terraform/downloads

Terraform state is stored in a shared GCS bucket. To initialize Terraform, run
the following:

```
# Authenticate with GCP
gcloud auth login --update-adc

# Initialize Terraform
terraform init
```

To preview your changes run `terraform plan`.

If the changes look correct, you can update the project with `terraform apply`.

Resources:

- Official Terraform documentation: https://developer.hashicorp.com/terraform/docs
- GCP Terraform documentation: https://cloud.google.com/docs/terraform/get-started-with-terraform
- Storing Terraform state in GCS: https://cloud.google.com/docs/terraform/resource-management/store-state
- Cloud Build Trigger documentation: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloudbuild_trigger
