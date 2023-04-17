module "cloud_build" {
  source = "../build_trigger"

  name                    = var.trigger_name
  github_repo             = "pytorch/xla"
  scheduler_account_email = var.scheduler_account_email
  timeout_minutes         = var.timeout_minutes
  worker_pool_id          = var.worker_pool_id
  description             = var.description
  location                = var.location

  trigger_on_push     = var.trigger_on_push
  trigger_on_schedule = var.trigger_on_schedule

  steps = concat(
    local.fetch_ansible_build_config,
    local.build_and_push_docker_image_steps,
    length(var.wheels_srcs) > 0 ? local.collect_and_publish_wheels_steps : []
  )
}

locals {
  fetch_ansible_build_config = [
    {
      id   = "git_fetch"
      name = "gcr.io/cloud-builders/git"
      args = ["fetch", "origin", var.ansible_branch]
    },
    {
      id   = "git_checkout"
      name = "gcr.io/cloud-builders/git"
      args = ["checkout", var.ansible_branch]
    }
  ]

  build_and_push_docker_image_steps = [
    # Build docker image.
    {
      id         = "build_${var.image_name}_docker_image"
      entrypoint = "bash"
      name       = "gcr.io/cloud-builders/docker"
      dir        = var.docker_context_dir
      args = [
        "-c",
        join(" ",
          concat(
            ["docker", "build", "--progress=plain", "--network=cloudbuild"],
            # Specify input docker file within the context.
            ["-f=${var.dockerfile}", "."],

            # Pass build args to the docker image.
            [for arg_key, arg_val in var.build_args : "--build-arg=${arg_key}=${arg_val}"],

            # Pass Ansible variables as JSON object, to make sure that
            # types are preserved (e.g. boolean),
            # see https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_variables.html#key-value-format.
            ["--build-arg=ansible_vars='${jsonencode(var.ansible_vars)}'"],

            # Pass all specified tags as $(echo <tag's bash expression>).
            # This allows to compute dynamic tags, e.g. date.
            [for tag in var.image_tags :
              "-t=\"${var.docker_repo_url}/${var.image_name}:$(echo ${tag})\""
            ],

            # Image used for the `copy_wheels_to_volume` step.
            ["-t=local_image"],
          )
        )
      ]
    },

    # Push all tags.
    {
      id         = "push_${var.image_name}_docker_image"
      entrypoint = "bash"
      name       = "gcr.io/cloud-builders/docker"
      args       = ["-c", "docker push --all-tags ${var.docker_repo_url}/${var.image_name}"]
    }
  ]

  wheels_srcs_str = join(" ", var.wheels_srcs)

  collect_and_publish_wheels_steps = [
    # Copy wheels from the local image to shared volume.
    {
      id         = "copy_wheels_to_volume"
      name       = "local_image"
      entrypoint = "bash"
      args = [
        "-c", join(" ",
          ["echo The following wheels will be published &&",
            "ls ${local.wheels_srcs_str} &&",
            "cp ${local.wheels_srcs_str} /wheels",
          ]
        )
      ]

      volumes = [{ name = "wheels", path = "/wheels" }]
    },

    # Upload collected wheels from volume to the storage bucket.
    {
      id         = "upload_wheels_to_storage_bucket"
      entrypoint = "bash"
      name       = "gcr.io/cloud-builders/gsutil"
      args = [
        "-c", "gsutil cp /wheels/*.whl ${var.wheels_dest}",
      ]

      volumes = [{ name = "wheels", path = "/wheels" }]
    },
  ]
}
