# Detailed documentation on cloudbuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build
resource "google_cloudbuild_trigger" "build-trigger" {
  location = "global"
  name = "dev-image-trigger"
  description = "Building docker development image"

  # Connect the repository in *global* region by going to
  # GCP Console > Triggers > Connect Repositiory.
  # Authorize and install the GCP App for the GitHub repositiry.
  github {
    owner = "mateuszlewko"
    name = "xla"
    push {
      branch = "^mlewko/cloudbuild$"
    }
  }

  source_to_build {
    uri = "https://github.com/mateuszlewko/xla"
    repo_type = "GITHUB"
    ref = "refs/heads/mlewko/cloudbuild"
  }

  included_files = [
    "docker/experimental/ansible/**",
    "docker/experimental/terraform_cloudbuild/**",
  ]

  build {
    # Build TPU Development image.
    step {
      id = "build_tpu_dev_image"
      name = "gcr.io/cloud-builders/docker"
      dir = "docker/experimental/ansible"
      args = [
        "build",
        "--build-arg=python_version=${var.python_version}",
        "-t=${var.image_repository}/development_tpu_amd64:latest",
        "-f=development.Dockerfile",
        ".",
      ]
      wait_for = [ "-" ] # Begin the step immediately.
      timeout = "${1 * 60 * 60}s" # 1h
    }

    # Build release image and wheels.
    step {
      id = "build_wheels_and_release_image"
      name = "gcr.io/cloud-builders/docker"
      dir = "docker/experimental/ansible"
      args = [
        "build",
        "--build-arg=python_version=${var.python_version}",
        "--build-arg=arch=amd64",
        "--build-arg=accelerator=tpu",
        "-t=${var.image_repository}/xla:latest",
        "-f=Dockerfile",
        ".",
      ]
      wait_for = [ "-" ] # Begin the step immediately.
      timeout = "${4 * 60 * 60}s" # 4h
    }

    artifacts {
      images = [
        "${var.image_repository}/development_tpu_amd64:latest",
        "${var.image_repository}/xla:latest",
      ]
    }

    options {
      substitution_option = "ALLOW_LOOSE"
      dynamic_substitutions = true
      # TODO: Variablize it
      worker_pool = "projects/core-ml-engprod-build-farm/locations/europe-west1/workerPools/compilerfarm"

    }

    timeout = "${6 * 60 * 60}s" # 6h
  }

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
}