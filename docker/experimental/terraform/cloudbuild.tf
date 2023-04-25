# Detailed documentation on cloudbuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build

locals {
  docker_images_map = {
    for di in var.docker_images :
    # Use either provided trigger name or image name and append triggers_suffix.
    "${coalesce(di.trigger_name, replace(di.image, "_", "-"))}${var.triggers_suffix}" => di
  }
}

resource "google_cloudbuild_trigger" "docker_images" {
  for_each = local.docker_images_map

  location    = "global"
  name        = each.key
  description = each.value.description

  dynamic "github" {
    # Trigger on branch push only if there is any `trigger_file` file filter.
    # To trigger on any push in the brunch set `trigger_files = ["**"]`
    for_each = length(each.value.trigger_files) > 0 ? [1] : []

    content {
      owner = "pytorch"
      name  = "xla"
      push {
        # `branch` is a regex, so look for exact match.
        branch = each.value.branch != "" ? "^${each.value.branch}$" : null
        tag    = each.value.git_tag != "" ? "^${each.value.git_tag}$" : null
      }
    }
  }

  source_to_build {
    uri       = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref       = each.value.branch != "" ? "refs/heads/${each.value.branch}" : "refs/tags/${each.value.git_tag}"
  }

  included_files = each.value.trigger_files

  build {
    step {
      id         = "build_${each.value.image}"
      entrypoint = "bash"
      name       = "gcr.io/cloud-builders/docker"
      dir        = each.value.dir
      args = [
        "-c",
        join(" ",
          concat(
            ["docker", "build", "--progress=plain", "--network=cloudbuild"],
            # Pass build args to the docker image.
            [for arg_key, arg_val in each.value.build_args : "--build-arg=${arg_key}=${arg_val}"],
            # Pass all specified tags as $(echo <tag's bash expression>).
            # This allows to compute dynamic tags, e.g. date.
            [for tag in each.value.image_tags :
              "-t=\"${local.public_docker_repo_url}/${each.value.image}:$(echo ${tag})\""
            ],
            # Image used for the `copy_wheels_to_volume` step.
            ["-t=local_image"],
            # Specify input docker file and context (current directory - each.value.dir)
            ["-f=${each.value.dockerfile}", "."]
          )
        )
      ]
    }

    step {
      id         = "push_${each.value.image}"
      entrypoint = "bash"
      name       = "gcr.io/cloud-builders/docker"
      args = [
        "-c", "docker push --all-tags ${local.public_docker_repo_url}/${each.value.image}"
      ]
    }

    dynamic "step" {
      for_each = each.value.wheels ? [1] : []

      content {
        # Copy wheels from the last built image to the shared volume.
        id         = "copy_wheels_to_volume"
        name       = "local_image"
        entrypoint = "bash"
        args = [
          "-c", join(" ",
            ["echo The following wheels will be published &&",
              "ls /dist/*.whl &&",
              "cp /dist/*.whl /wheels",
            ]
          )
        ]

        volumes {
          name = "wheels"
          path = "/wheels"
        }
      }
    }

    dynamic "step" {
      for_each = each.value.wheels ? [1] : []

      content {
        # Upload copied images from the shared volume to the public storage bucket.
        id         = "upload_wheels_to_storage_bucket"
        entrypoint = "bash"
        name       = "gcr.io/cloud-builders/gsutil"
        args = [
          "-c", "gsutil cp /wheels/*.whl ${google_storage_bucket.public_wheels.url}",
        ]

        volumes {
          name = "wheels"
          path = "/wheels"
        }
      }
    }

    options {
      substitution_option   = "ALLOW_LOOSE"
      dynamic_substitutions = true
      worker_pool           = local.worker_pool_id
    }

    timeout = "${each.value.timeout_m * 60}s"
  }

  include_build_logs = length(each.value.trigger_files) > 0 ? "INCLUDE_BUILD_LOGS_WITH_STATUS" : null
}

# Add scheduled jobs for each cloudbuild_trigger with a schedule.
module "schedule_triggers" {
  source = "./modules/trigger_schedule"
  for_each = {
    for name, di in local.docker_images_map : name => di if di.trigger_schedule != ""
  }

  trigger                   = google_cloudbuild_trigger.docker_images[each.key]
  schedule                  = each.value.trigger_schedule
  scheduler_service_account = google_service_account.build_runner.email
}
