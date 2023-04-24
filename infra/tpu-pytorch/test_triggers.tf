module "tpu_e2e_tests" {
  source = "../terraform_modules/xla_docker_build"

  trigger_name = "ci-tpu-test-trigger2"

  ansible_branch  = "mlewko/ci-tpu-test-trigger"
  trigger_on_push = { branch = "master" }
  run_e2e_tests   = true

  image_name = "pytorch-xla-test"
  image_tags = [
    # $BUILD_ID is a GCB variable, not a bash variable.
    # See https://cloud.google.com/build/docs/configuring-builds/substitute-variable-values#using_default_substitutions.
    "$BUILD_ID",
  ]
  dockerfile = "e2e_tests.Dockerfile"
  description = join(" ", [
    "Run e2e TPU tests on an image built from master branch.",
    "Trigger managed by Terraform setup in",
    "infra/tpu-pytorch/test_triggers.tf.",
  ])

  build_args = {
    python_version = "3.8"
    debian_version = "buster"
  }

  ansible_vars = {
    arch            = "amd64"
    accelerator     = "tpu"
    pytorch_git_rev = "HEAD"
    xla_git_rev     = "HEAD"
  }

  # Substitutions used in the "run_e2e_tests" step, see
  # infra/terraform_modules/xla_docker_build/xla_docker_build.tf.
  substitutions = {
    _CLUSTER_NAME = "tpu-cluster"
    _CLUSTER_ZONE = "europe-west4-a"
  }

  docker_repo_url = module.docker_registry.url
  worker_pool_id  = module.worker_pool.id
  timeout_minutes = 4 * 60
  location        = "global"
}
