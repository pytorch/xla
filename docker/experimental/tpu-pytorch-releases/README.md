# Terraform setup for the `pytorch-xla-releases` GCP project

This setup configures all resources for building public artifacts: docker images 
and python wheels.

## Cloud Build Triggers

This section explains how to add, modify and run Cloud Build triggers to:
* add new releases,
* modify existing releases or nightly builds,
* remove old releases.

The list of Cloud Build triggers is specified in the 
[artifacts.auto.tfvars](./artifacts.auto.tfvars) file, in two variables 
`versioned_builds` and `nightly_builds`.

These variables are consumed in the [cloud_builds.tf](./cloud_builds.tf) file.

Each build is associated with a separate build trigger.
Build trigger builds both docker image and Python wheels.

* Docker images are pushed to the configured docker registry: 
[us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla](http://us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla)
* Python wheels are uploaded to `gs://pytorch-xla-releases/wheels/` 
([GCP link](https://pantheon.corp.google.com/storage/browser/pytorch-xla-releases/wheels)).

### Versioned releases

Versioned release builds are triggered on push to a specific `git_tag`.

Versioned release entries in the `versioned_builds` variable in 
[artifacts.auto.tfvars](./artifacts.auto.tfvars)
consists of the following fields.
* `git_tag` (string) - Git tag at which to checkout both PyTorch and PyTorch/XLA 
  sources when building image and wheels.
* `package_version` (string) - Version of the built wheels. Passed to the 
  build steps.
* `accelerator` ("tpu"|"cuda") - Supported accelerator. Impacts build
  process and installed dependencies.
* `python_version` (optional, string, default = "3.8") - Python version used for 
  the docker images base and build process.
* `cuda_version` (optional, string, default = "11.8") - CUDA version to install.
  Used only if `accelerator` is set to "cuda"
* `arch` (optional, "amd64"|"aarch64", default = "amd64") - Architecture 
  influences installed dependencies and build process.

To modify default values see `variable "versioned_builds"` in 
[cloud_builds.tf](./cloud_builds.tf). Modifying default values will modify 
unset properties of existing triggers.

#### Add a new versioned release

1. Add an entry with specific git tag, accelerator, package and python versions 
to the `versioned_builds` variable in the 
[artifacts.auto.tfvars](./artifacts.auto.tfvars) file. 
See all variables in the section above.
  
    **Example**

    ```hcl
    versioned_builds = [
      {
        git_tag         = "v3.0.0"
        package_version = "3.0"
        accelerator     = "tpu"
        cuda_version    = "11.8"  # optional
        python_version  = "3.8"   # optional
        arch            = "amd64" # optional
      },
      # ...
    ]
    ```
2. Ensure that Terraform repo is initialized, see 
   [Terraform.md](../Terraform.md).
3. Run `terraform apply` and review the planned changes.
4. Types "yes" to confirm the changes. Wait for Terraform to enforce all 
   changes.
5. (Optional) See section "Manually trigger a Cloud Build" to manually trigger 
   the created build.


### Nightly releases

Nightly release are configured to build from the `master` branch once per day 
at midnight (America/Los_Angeles time zone). 

Nightly builds in the `nightly_builds` variable in 
[artifacts.auto.tfvars](./artifacts.auto.tfvars)
consists of the following fields.
* `accelerator` ("tpu"|"cuda") - Supported accelerator. Impacts build
  process and installed dependencies.
* `python_version` (optional, string, default = "3.8") - Python version used for 
  the docker images base and build process.
* `cuda_version` (optional, string, default = "11.8") - CUDA version to install.
  Used only if `accelerator` is set to "cuda"
* `arch` (optional, "amd64"|"aarch64", default = "amd64") - Architecture 
  influences installed dependencies and build process.

Additionally, **`package_version` of all nightly builds** is configured through 
a separate `nightly_package_version` variable.

To modify default values see `variable "nightly_builds"` in 
[cloud_builds.tf](./cloud_builds.tf). Modifying default values will modify 
unset properties of existing triggers.

#### Add a new nightly release

1. Add an entry with specific accelerator, python and (optionally) cuda version 
to the `nightly_builds` variable in the 
[artifacts.auto.tfvars](./artifacts.auto.tfvars) file.
See all variables in the section above.
  
    **Example**

    ```hcl
    nightly_builds = [
      {
        accelerator    = "cuda"
        cuda_version   = "11.8"  # optional
        python_version = "3.8" # optional
        arch           = "amd64" # optional
      },
      # ...
    ]
    ```
2. Ensure that Terraform repo is initialized, see 
   [Terraform.md](../Terraform.md).
3. Run `terraform apply` and review the planned changes.
4. Types "yes" to confirm the changes. Wait for Terraform to enforce all 
   changes.
5. (Optional) See section "Manually trigger a Cloud Build" to manually trigger 
   the created build. Nightly build will be triggered automatically at midnight.

### Manually trigger a Cloud Build

1. Go to [Cloud Build > Triggers](https://pantheon.corp.google.com/cloud-build/triggers;region=us-central1?project=tpu-pytorch-releases) page in GCP.
2. Click "RUN" on the desired triggered. 
   
   **Note:** "Branch" input in the "Run trigger" window is irrelevant, since 
   Ansible setup and repository sources will be fetched at revisions specified 
   in [cloud_builds.tf](./cloud_builds.tf).

3. Click "Run Trigger"
4. Go to [History] (https://pantheon.corp.google.com/cloud-build/builds;region=us-central1?project=tpu-pytorch-releases)
   to see status of the triggered builds.
