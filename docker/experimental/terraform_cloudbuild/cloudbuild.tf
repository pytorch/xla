# Detailed documentation on cloudbuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build
resource "google_cloudbuild_trigger" "build-trigger" {
  location = "global"
  name = "dev-image-trigger"

  # Connect the repository in *global* region by going to
  # GCP Console > Triggers > Connect Repositiory.
  # Authorize and install the GCP App for the GitHub repositiry.
  github {
    owner = "mateuszlewko"
    name = "xla"
    push {
      branch = "^cloudbuild$"
    }
  }

  source_to_build {
    uri = "https://github.com/mateuszlewko/xla"
    repo_type = "GITHUB"
    ref = "refs/heads/cloudbuild"
  }

  included_files = [
    "docker/experimental/ansible/**",
    "docker/experimental/terraform_cloudbuild/**",
  ]

  build {
    # Build TPU Development image.
    step {
      name = "gcr.io/cloud-builders/docker"
      dir = "docker/experimental/ansible"
      args = [
        "build",
        "--build-arg=python_version=${var.python_version}",
        "-t=${var.image_repository}/development_tpu_amd64:latest",
        "-f=development.Dockerfile",
        ".",
      ]
      timeout = "${1 * 60 * 60}s" # 1h
    }

    artifacts {
      images = [
        "${var.image_repository}/development_tpu_amd64:latest",
      ]
    }

    options {
      substitution_option = "ALLOW_LOOSE"
      dynamic_substitutions = true
      # TODO: Variablize it
      worker_pool = "projects/core-ml-engprod-build-farm/locations/europe-west1/workerPools/compilerfarm"

    }

    timeout = "${5 * 60 * 60}s" # 5h
  }

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
}