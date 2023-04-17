# Terraform

This file contains instructions for common Terraform operations.

## Install Terraform

Make sure that a recent Terraform binary is installed (>= 1.3.8).
If not, install Terraform from the [official source](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).

## First time initialization

1. Run `gcloud auth application-default login` on your local workstation.
2. Go to the directory of the desired Terraform setup, for example 
   [`tpu-pytorch-releases`](./tpu-pytorch-releases).
3. Run `terraform init`.

## Enforce entire Terraform setup

1. Run `terraform apply`. Preview the planned changes.
2. Confirm planned changes by typing "yes" and pressing enter.
3. Wait for Terraform to finish provisioning resources.

## Enforce only selected resource.

1. Run `terraform apply` to preview planned changes.
2. Note the Terraform resource ID of the resource that you want to provision.

  **Example**

  For the following Terraform plan:
  ```
  Terraform will perform the following actions:

    # module.bazel_builds.module.cloud_build.google_cloudbuild_trigger.trigger will be destroyed
    # (because google_cloudbuild_trigger.trigger is not in configuration)
    - resource "google_cloudbuild_trigger" "trigger" {
        - create_time        = "2023-04-13T10:52:58.971939642Z" -> null
   
   ...
   ```

   `module.bazel_builds.module.cloud_build.google_cloudbuild_trigger.trigger` is 
   a resource ID.

3. Run `terraform apply -target=$RESOURCE_ID`. 
   Verify that only the desired resource will be modified. 
   The flag can be used multiple times, also with `terraform destroy` command.

## Check if Terraform setup is fully provisioned

1. Running `terraform plan` should return empty plan if local configuration was 
   fully provisioned. 
   Terraform won't show diff in any resources that were not created by Terraform.