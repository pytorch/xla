# Detailed documentation on cloudbuild parameters:
# https://cloud.google.com/build/docs/api/reference/rest/v1/projects.builds#resource-build


locals {
  docker_images_map = {
    for di in var.docker_images:
      # Use either provided name or use "<image_name>-trigger".
      coalesce(di.name, "${replace(di.image, "_", "-")}-trigger${var.triggers_suffix}") => di
  }
}

resource "google_cloudbuild_trigger" "docker_images" {
  for_each = local.docker_images_map

  location = "global"
  name = coalesce(
    each.value.name,
     "${replace(each.value.image, "_", "-")}-trigger${var.triggers_suffix}"
  )
  description = each.value.description

  dynamic "github" {
    # Trigger on branch push only if there is any `trigger_file` file filter.
    # To trigger on any push in the brunch set `trigger_files = ["**"]`
    for_each = length(each.value.trigger_files) > 0 ? [1] : []

    content {
      owner = "pytorch"
      name = "xla"
      push {
        # `branch` is a regex, so look for exact match.
        branch = each.value.branch != "" ? "^${each.value.branch}$" : null
        tag = each.value.git_tag != "" ? "^${each.value.git_tag}$" : null
      }
    }
  }

  source_to_build {
    uri = "https://github.com/pytorch/xla"
    repo_type = "GITHUB"
    ref = each.value.branch != "" ? "refs/heads/${each.value.branch}" : "refs/tags/${each.value.git_tag}"
  }

  included_files = each.value.trigger_files

  build {
     step {
      id = "build_${each.value.image}"
      entrypoint = "bash"
      name = "gcr.io/cloud-builders/docker"
      dir = each.value.dir
      args = [
        "-c",
        join(" ",
          concat(
            ["docker", "build" ],
            [ for arg_key, arg_val in each.value.build_args: "--build-arg=${arg_key}=${arg_val}" ],
            [ for tag in each.value.image_tags: "-t=\"${local.public_docker_repo_url}/${each.value.image}:$(echo ${tag})\""],
            ["-f=${each.value.dockerfile}", "."]
          )
        )
      ]
    }

   step {
      id = "push_${each.value.image}"
      entrypoint = "bash"
      name = "gcr.io/cloud-builders/docker"
      args = [
        "-c", "docker push --all-tags ${local.public_docker_repo_url}/${each.value.image}"
      ]
    }

    artifacts {
      objects {
        location = "${google_storage_bucket.public_wheels.url}"
        paths = each.value.wheels
      }
    }

    options {
      substitution_option = "ALLOW_LOOSE"
      dynamic_substitutions = true
      worker_pool = local.worker_pool_id
    }

    timeout = "${each.value.timeout_m * 60}s"
  }
}

# Add scheduled jobs for each cloudbuild_trigger with a schedule.
module "schedule_triggers" {
  source = "./modules/trigger_schedule"
  for_each = {
    for name, di in local.docker_images_map: name => di if di.trigger_schedule != ""
  }

  trigger = google_cloudbuild_trigger.docker_images[each.key]
  schedule = each.value.trigger_schedule
  scheduler_service_account = google_service_account.build_runner.email
}

# resource "google_cloudbuild_trigger" "release_images" {
#   location = "global"
#   name = "release-image-trigger"
#   description = "Building docker release image"

#   source_to_build {
#     uri = "https://github.com/pytorch/xla"
#     repo_type = "GITHUB"
#     ref = "refs/heads/mlewko/cloudbuild"
#   }

#   build {
#     step {
#       id = "build_cuda_release_image"
#       name = "gcr.io/cloud-builders/docker"
#       dir = "docker/experimental/ansible"
#       args = [
#         "build",
#         "--build-arg=python_version=${var.python_version}",
#         "--build-arg=accelerator=cuda",
#         # TODO Variablize Cuda version
#         "-t=${local.public_docker_repo_url}/xla:nightly_${var.python_version}_cuda_11.8",
#         "-f=Dockerfile",
#         ".",
#       ]
#       wait_for = [ "-" ] # Begin the step immediately.
#       timeout = "${5 * 60 * 60}s"
#     }

#     step {
#       id = "build_tpu_release_image"
#       name = "gcr.io/cloud-builders/docker"
#       dir = "docker/experimental/ansible"
#       args = [
#         "build",
#         "--build-arg=python_version=${var.python_version}",
#         "--build-arg=accelerator=tpu",
#         "-t=${local.public_docker_repo_url}/xla:nightly_${var.python_version}_tpuvm",
#         "-f=Dockerfile",
#         ".",
#       ]
#       wait_for = [ "-" ] # Begin the step immediately.
#       timeout = "${5 * 60 * 60}s"
#     }

#     artifacts {
#       images = [
#         "${local.public_docker_repo_url}/xla:nightly_${var.python_version}_cuda_11.8",
#         "${local.public_docker_repo_url}/xla:nightly_${var.python_version}_tpuvm",
#       ]
#       objects {
#         location = "${google_storage_bucket.public_wheels.url}"
#         paths = [
#           "/wheels/*.whl"
#         ]
#       }
#     }
#     options {
#       substitution_option = "ALLOW_LOOSE"
#       dynamic_substitutions = true
#       worker_pool = local.worker_pool_id
#     }

#     timeout = "${10 * 60 * 60}s"
#   }
# }

# module "release_images_trigger" {
#   source = "./modules/trigger_schedule"
#   trigger = google_cloudbuild_trigger.release_images
#   scheduler_service_account = google_service_account.build_runner.email
# }

# output "triggers" {
#   value = [
#     # google_cloudbuild_trigger.dev-image.id,
#     google_cloudbuild_trigger.release_images.id,
#   ]
# }