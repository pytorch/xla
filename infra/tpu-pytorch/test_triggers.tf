resource "google_cloudbuild_trigger" "tpu-test-trigger" {
  name = "ci-tpu-test-trigger"

  github {
    owner = "pytorch"
    name  = "xla"
    push {
      branch = "^master$"
    }
  }

  description = "Trigger managed by Terraform setup in infra/tpu-pytorch/test_triggers.tf."
  filename    = "test/tpu/cloudbuild.yaml"
}
