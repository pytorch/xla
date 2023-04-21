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
    length(var.wheels_srcs) > 0 ? local.collect_and_publish_wheels_steps : [],
    var.run_e2e_tests ? local.e2e_tests_steps : [],
  )

  substitutions = var.substitutions
}

locals {
  fetch_ansible_build_config = [
    {
      id   = "git_fetch"
      name = "gcr.io/cloud-builders/git"
      args = ["fetch", "--unshallow"]
    },
    {
      id   = "git_checkout"
      name = "gcr.io/cloud-builders/git"
      args = ["checkout", "origin/${var.ansible_branch}"]
    },
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

  # Runs CI end2end tests on k8s.
  e2e_tests_steps = [
    {
      id         = "run_e2e_tests"
      name       = "google/cloud-sdk"
      entrypoint = "bash"
      args = [
        "-c", <<EOT
          set -u # Fail if any variables are unset.
          set -e # Exit immediately if any command fails.
          set -x # Print executed commands.

          apt-get update
          apt-get -y install gettext

          # Try to setup kubectl credentials the new way,
          # see https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
          apt-get install google-cloud-sdk-gke-gcloud-auth-plugin -o DPkg::options::="--force-overwrite"
          export USE_GKE_GCLOUD_AUTH_PLUGIN=True
          apt-get install kubectl
          gcloud container clusters get-credentials $_CLUSTER_NAME --zone $_CLUSTER_ZONE

          # Export variables to shell environment so that we can pass them into the
          # k8s deployment using envsubst
          export IMAGE=${var.docker_repo_url}/${var.image_name}:${var.image_tags[0]}
          export PROJECT_ID=$PROJECT_ID

          # Launch k8s deployment, wait for completion, print logs
          pod_name=$(envsubst < test/tpu/xla_test_job.yaml | kubectl create -f - -o name)
          pod_name=$(kubectl wait --for condition=ready --timeout=10m $pod_name -o name)
          kubectl logs -f $pod_name --container=xla-test

          exit $(kubectl get $pod_name -o jsonpath='{.status.containerStatuses[?(@.name=="xla-test")].state.terminated.exitCode}')
        EOT
      ]
    }
  ]
}
