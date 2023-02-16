# Detailed documentation on cloudbuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build

resource "google_cloudbuild_trigger" "docker_images" {
  count = length(var.docker_images)

  location = "global"
  name = "${replace(var.docker_images[count.index].image, "_", "-")}-trigger${var.triggers_suffix}"
  description = var.docker_images[count.index].description

  dynamic "github" {
    # Trigger on branch push only if there is any `trigger_file` file filter.
    # To trigger on any push in the brunch set `trigger_files = ["**"]`
    for_each = length(var.docker_images[count.index].trigger_files) > 0 ? [1] : []

    content {
      owner = "pytorch"
      name = "xla"
      push {
        # `branch` is a regex, so look for exact match.
        branch = var.docker_images[count.index].branch != "" ? "^${var.docker_images[count.index].branch}$" : null
        tag = var.docker_images[count.index].git_tag != "" ? "^${var.docker_images[count.index].git_tag}$" : null
      }
    }
  }

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref = var.docker_images[count.index].branch != "" ? "refs/heads/${var.docker_images[count.index].branch}" : "refs/tags/${var.docker_images[count.index].git_tag}"
  }

  included_files = var.docker_images[count.index].trigger_files

  build {
     step {
      id = "build_${var.docker_images[count.index].image}"
      entrypoint = "bash"
      name = "gcr.io/cloud-builders/docker"
      args = concat(
        [ "docker", "build" ],
        [ for arg in var.docker_images[count.index].build_args: "--build-arg=${arg}" ],
        [ for tag in var.docker_images[count.index].image_tags: "-t=\"${local.public_docker_repo_url}/${var.docker_images[count.index].image}:$(echo ${tag})\""],
        ["-f=${var.docker_images[count.index].dockerfile}", "."]
      )
    }

   step {
      id = "push_${var.docker_images[count.index].image}"
      entrypoint = "bash"
      name = "gcr.io/cloud-builders/docker"
      args = [
        "docker", "push", "--all-tags", "${local.public_docker_repo_url}/${var.docker_images[count.index].image}"
      ]
    }

    options {
      substitution_option = "ALLOW_LOOSE"
      dynamic_substitutions = true
      worker_pool = local.worker_pool_id
    }

    timeout = "${var.docker_images[count.index].timeout_m * 60}s"
  }
}

# resource "google_cloudbuild_trigger" "dev-image" {
#   location = "global"
#   name = "dev-image-trigger"
#   description = "Building docker development image"

#   # Connect the repository in *global* region by going to
#   # GCP Console > Triggers > Connect Repository.
#   # Authorize and install the GCP App for the GitHub repository.
#   github {
#     owner = "pytorch"
#     name = "xla"
#     push {
#       branch = "^mlewko/cloudbuild$"
#     }
#   }

#   source_to_build {
#     uri = "https://github.com/pytorch/xla"
#     repo_type = "GITHUB"
#     ref = "refs/heads/mlewko/cloudbuild"
#   }

#   included_files = [
#     "docker/experimental/ansible/**",
#     "docker/experimental/terraform_cloudbuild/**",
#   ]

#   build {
#     # Build TPU Development image.
#     step {
#       id = "build_tpu_dev_image"
#       name = "gcr.io/cloud-builders/docker"
#       dir = "docker/experimental/ansible"
#       args = [
#         "build",
#         "--build-arg=python_version=${var.python_version}",
#         "-t=${local.public_docker_repo_url}/development_tpu_amd64:latest",
#         "-f=development.Dockerfile",
#         ".",
#       ]
#       wait_for = [ "-" ] # Begin the step immediately.
#       timeout = "${1 * 60 * 60}s" # 1h
#     }

#     artifacts {
#       images = [
#         "${local.public_docker_repo_url}/development_tpu_amd64:latest",
#       ]
#     }

#     options {
#       substitution_option = "ALLOW_LOOSE"
#       dynamic_substitutions = true
#       worker_pool = local.worker_pool_id
#     }

#     timeout = "${6 * 60 * 60}s" # 6h
#   }

#   include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
# }


resource "google_cloudbuild_trigger" "release_images" {
  location = "global"
  name = "release-image-trigger"
  description = "Building docker release image"

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref = "refs/heads/mlewko/cloudbuild"
  }

  build {
    step {
      id = "build_cuda_release_image"
      name = "gcr.io/cloud-builders/docker"
      dir = "docker/experimental/ansible"
      args = [
        "build",
        "--build-arg=python_version=${var.python_version}",
        "--build-arg=accelerator=cuda",
        # TODO Variablize Cuda version
        "-t=${local.public_docker_repo_url}/xla:nightly_${var.python_version}_cuda_11.8",
        "-f=Dockerfile",
        ".",
      ]
      wait_for = [ "-" ] # Begin the step immediately.
      timeout = "${5 * 60 * 60}s"
    }

    step {
      id = "build_tpu_release_image"
      name = "gcr.io/cloud-builders/docker"
      dir = "docker/experimental/ansible"
      args = [
        "build",
        "--build-arg=python_version=${var.python_version}",
        "--build-arg=accelerator=tpu",
        "-t=${local.public_docker_repo_url}/xla:nightly_${var.python_version}_tpuvm",
        "-f=Dockerfile",
        ".",
      ]
      wait_for = [ "-" ] # Begin the step immediately.
      timeout = "${5 * 60 * 60}s"
    }

    artifacts {
      images = [
        "${local.public_docker_repo_url}/xla:nightly_${var.python_version}_cuda_11.8",
        "${local.public_docker_repo_url}/xla:nightly_${var.python_version}_tpuvm",
      ]
      objects {
        location = "${google_storage_bucket.public_wheels.url}"
        paths = [
          "/wheels/*.whl"
        ]
      }
    }
    options {
      substitution_option = "ALLOW_LOOSE"
      dynamic_substitutions = true
      worker_pool = local.worker_pool_id
    }

    timeout = "${10 * 60 * 60}s"
  }
}

module "release_images_trigger" {
  source = "./modules/trigger_schedule"
  trigger = google_cloudbuild_trigger.release_images
  scheduler_service_account = google_service_account.build_runner.email
}

output "triggers" {
  value = [
    # google_cloudbuild_trigger.dev-image.id,
    google_cloudbuild_trigger.release_images.id,
  ]
}