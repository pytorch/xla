provider "google" {
  project = var.project_id
}

provider "helm" {
  kubernetes {
    host = "https://${google_container_cluster.arc_v4_cluster.endpoint}"
    token = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.arc_v4_cluster.master_auth.0.cluster_ca_certificate)
  }
}

data "google_client_config" "default" {}

resource "google_container_cluster" "arc_v4_cluster" {
  name = var.cluster_name
  location = "us-central2"

  remove_default_node_pool = true
  initial_node_count = 1

  release_channel {
    channel = "RAPID"
  }

  min_master_version = 1.28
}

resource "google_container_node_pool" "arc_v4_cpu_nodes" {
  name = var.cpu_nodepool_name
  location = "us-central2"
  cluster = google_container_cluster.arc_v4_cluster.name
  node_count = var.cpu_node_count

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]

    machine_type = "n1-standard-1"
  }

  management {
    auto_upgrade = true
    auto_repair = true
  }
}

resource "google_container_node_pool" "arc_v4_tpu_nodes" {
  name = var.tpu_nodepool_name
  location = "us-central2"
  node_locations = ["us-central2-b"]
  cluster = google_container_cluster.arc_v4_cluster.name
  initial_node_count = 0
  autoscaling {
    total_min_node_count = 0
    total_max_node_count = var.max_tpu_nodes
    location_policy = "ANY"
  }
  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]
    machine_type = "ct4p-hightpu-4t"
  }
  management {
    auto_upgrade = true
    auto_repair = true
  }
}

resource "helm_release" "arc" {
  name = "actions-runner-controller"
  chart = "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller"
  namespace = var.arc_namespace
  create_namespace = true
}

resource "helm_release" "arc_runner_set" {
  name = "v4-runner-set"
  depends_on = [
    helm_release.arc
  ]
  chart = "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set"
  namespace = var.runner_namespace
  create_namespace = true

  values = [
    templatefile("modules/google-arc-v4-container-cluster/arc-values.yaml", {
      github_repo_url = var.github_repo_url
      max_tpu_nodes = var.max_tpu_nodes
      runner_image = var.runner_image
    })
  ]
}
