# Terraform setup for the `pytorch-xla-releases` GCP project

This setup configures all resources for building public artifacts: docker images
and python wheels.

In addition to public artifacts, the setup also configures:
* Cloud Build trigger for the public development Docker image (`dev-image`).


## Cloud Build Triggers

This section explains how to add, modify and run Cloud Build triggers to:
* add new releases,
* modify existing releases or nightly builds,
* remove old releases.

The list of Cloud Build triggers is specified in the
[artifacts.auto.tfvars](./artifacts.auto.tfvars) file, in two variables
`versioned_builds` and `nightly_builds`.

These variables are consumed in the [artifacts_builds.tf](./artifacts_builds.tf) file.

Each artifact is associated with a separate build trigger.
A build trigger builds both docker image and Python wheels.

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
* `accelerator` ("tpu"|"cuda") - Supported accelerator. Affects build
  process and installed dependencies, see [apt.yaml](../ansible/config/apt.yaml) and
  [pip.yaml](../ansible/config/pip.yaml).
* `python_version` (optional, string, default = "3.8") - Python version used for
  the docker image base and build process.
* `cuda_version` (optional, string, default = "11.8") - CUDA version to install.
  Used only if `accelerator` is set to "cuda"
* `arch` (optional, "amd64"|"aarch64", default = "amd64") - Architecture
  affects installed dependencies and build process, see [apt.yaml](../ansible/config/apt.yaml) and
  [pip.yaml](../ansible/config/pip.yaml).
* `cxx11_abi` (optional, "0"|"1", default = "0") - Whether to use C++11 ABI or
  pre-C++11 ABI.

To modify default values see `variable "versioned_builds"` in
[artifacts_builds.tf](./artifacts_builds.tf). Modifying default values will modify
unset properties of existing triggers.

#### Add a new versioned release

1. Add an entry with specific `git_tag`, `accelerator`, `package_version` and
   `python_version` to the `versioned_builds` variable in the
   [artifacts.auto.tfvars](./artifacts.auto.tfvars) file.
   See all variables definitions in the section above.

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
2. See [Preview Terraform changes](https://github.com/pytorch/xla/blob/master/infra/Terraform.md#preview-terraform-changes)
   to preview proposed Terraform changes without affecting any infrastructure.
3. Commit proposed changes.
4. After successfully merge, [`terraform-provision-trigger`](https://pantheon.corp.google.com/cloud-build/builds;region=us-central1?project=tpu-pytorch-releases&pageState=(%22builds%22:(%22f%22:%22%255B%257B_22k_22_3A_22Trigger%2520Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22terraform-provision-trigger_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22triggerName_22%257D%255D%22)))
   will run and enforce all the proposed infrastructure changes.
5. See section [Manually trigger a Cloud Build](#manually-trigger-a-cloud-build)
   to manually trigger the created build and produce all the artifacts.


### Nightly releases

Nightly release are configured to build from the `master` branch once per day
at midnight (`America/Los_Angeles` time zone).

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
* `cxx11_abi` (optional, "0"|"1", default = "0") - Whether to use C++11 ABI or
  pre-C++11 ABI.

Additionally, **`package_version` of all nightly builds** is configured through
a separate `nightly_package_version` variable.

To modify default values see `variable "nightly_builds"` in
[artifacts_builds.tf](./artifacts_builds.tf). Modifying default values will modify
unset properties of existing triggers.

#### Modify or add a new nightly release

1. Modify or add an entry with specific `accelerator`, `python_version` and (optionally)
   `cuda_version` to the `nightly_builds` variable in the
   [artifacts.auto.tfvars](./artifacts.auto.tfvars) file.
   See all variables in the section above.

    **Example**

    ```hcl
    nightly_builds = [
      {
        accelerator    = "cuda"
        cuda_version   = "11.8"  # optional
        python_version = "3.8"   # optional
        arch           = "amd64" # optional
      },
      # ...
    ]
    ```
2. See [Preview Terraform changes](https://github.com/pytorch/xla/blob/master/infra/Terraform.md#preview-terraform-changes)
   to preview proposed Terraform changes without affecting any infrastructure.
3. Commit proposed changes.
4. After successfully merge, [`terraform-provision-trigger`](https://pantheon.corp.google.com/cloud-build/builds;region=us-central1?project=tpu-pytorch-releases&pageState=(%22builds%22:(%22f%22:%22%255B%257B_22k_22_3A_22Trigger%2520Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22terraform-provision-trigger_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22triggerName_22%257D%255D%22)))
   will run and enforce all the proposed infrastructure changes.
5. See section [Manually trigger a Cloud Build](#manually-trigger-a-cloud-build)
   to manually trigger the created build and produce all the artifacts.


### Manually trigger a Cloud Build

1. Go to [Cloud Build > Triggers](https://pantheon.corp.google.com/cloud-build/triggers;region=us-central1?project=tpu-pytorch-releases) page in GCP.
2. Click "RUN" on the desired triggered.

   **Note:** "Branch" input in the "Run trigger" window is irrelevant, since
   Ansible setup and repository sources will be fetched at revisions specified
   in [artifacts_builds.tf](./artifacts_builds.tf).

3. Click "Run Trigger"
4. Go to [History](https://pantheon.corp.google.com/cloud-build/builds;region=us-central1?project=tpu-pytorch-releases)
   to see status of the triggered builds.


### Create experimental trigger for non-master branch

1. Add a new instance of `xla_docker_build` module to [artifacts_builds.tf](./artifacts_builds.tf)
   (or any other or new file within that directory - Terraform reads automatically all top-level
   files from the setup directory).

    **Example**

    ```hcl
    module "my_branch" {
      source   = "../terraform_modules/xla_docker_build"

      ansible_vars = merge(each.value, {
        pytorch_git_rev = "main"
        # Fetch XLA sources from "my-branch".
        # You can also use any git revision (e.g. tag), or
        # "$COMMIT_SHA" to fetch the sources at the same commit that
        # the Build was triggered.
        xla_git_rev     = "my-branch"
      })

      # Fetch Ansible configuration from "my-branch".
      ansible_git_rev  = "my-branch"

      # Build will be triggered on every push to "my-branch".
      trigger_on_push = {
        branch = "my-branch"
      }

      # Trigger name in GCP.
      trigger_name = "trigger-for-my-branch"

      # Remove `image_name` and `image_tags` if you don't want to
      # upload any docker images
      image_name   = "my-experimental-image"
      image_tags   = ["$COMMIT_ID", "latest"]

      description = "Experimental trigger for my-branch"

      # Remove `wheels_dest` and `wheels_srcs` if you don't want to
      # upload any Python wheels.
      wheels_dest = "${module.releases_storage_bucket.url}/wheels/experimental/my-branch-name"
      wheels_srcs = ["/dist/*.whl"]

      # Passed directly to ../ansible/Dockerfile.
      build_args  = {
        python_version = "3.8" # Default, can be removed.
      }

      worker_pool_id  = module.worker_pool.id

      # Remove or change to a different docker registry.
      docker_repo_url = module.docker_registry.url
    }
    ```

2. Create the trigger in GCP. Complete either of the steps below.

    a) Either commit and merge the changed Terraform setup
       to master to get it automatically applied, or

    b) apply manually only the newly created
       resource, see
       [Enforce only selected resource](https://github.com/pytorch/xla/blob/master/infra/Terraform.md#enforce-only-selected-resource) (this requires appropriate permissions in GCP).
