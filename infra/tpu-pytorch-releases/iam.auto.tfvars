project_admins = [
  "group:cloud-tpus-grpadm@twosync.google.com",
  "group:pytorchxla-dev@google.com",
]

cloudbuild_editors = [
]

project_remote_build_writers = [
  "group:cloud-tpus-dev-team@twosync.google.com",
  "user:pytorchxla-general@google.com",
  # tpu-pytorch-releases project: default Service Account for running Cloud Build jobs.
  "serviceAccount:1001674285173@cloudbuild.gserviceaccount.com"
]
