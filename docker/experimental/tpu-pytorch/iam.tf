resource "google_project_iam_custom_role" "remote_bazel_role" {
  role_id     = "remoteBuildWriterRole"
  title       = "Remote Build Writer"
  description = "For running remote bazel builds and read/write from remote cache on GCP."
  permissions = [
    "remotebuildexecution.actions.create",
    "remotebuildexecution.actions.get",
    "remotebuildexecution.actions.set",
    "remotebuildexecution.blobs.create",
    "remotebuildexecution.blobs.get",
    "remotebuildexecution.logstreams.create",
    "remotebuildexecution.logstreams.get",
    "remotebuildexecution.logstreams.update",
  ]
}
