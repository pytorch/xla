provider "google" {
  project = var.project_id
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.arc_v6e_cluster.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.arc_v6e_cluster.master_auth.0.cluster_ca_certificate)
  }
}

data "google_client_config" "default" {}

resource "google_container_cluster" "arc_v6e_cluster" {
  name     = var.cluster_name
  location = "us-central2"

  remove_default_node_pool = true
  initial_node_count       = 1

  release_channel {
    channel = "RAPID"
  }

  min_master_version = 1.28
}

resource "google_container_node_pool" "arc_v6e_cpu_nodes" {
  name       = var.cpu_nodepool_name
  location   = "us-central2"
  cluster    = google_container_cluster.arc_v6e_cluster.name
  node_count = var.cpu_node_count

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]
  }

  management {
    auto_upgrade = true
    auto_repair  = true
  }
}

resource "google_container_node_pool" "arc_v6e_tpu_nodes" {
  name               = var.tpu_nodepool_name
  location           = "us-central2"
  node_locations     = ["us-central2-b"]
  cluster            = google_container_cluster.arc_v6e_cluster.name
  initial_node_count = 1
  autoscaling {
    total_min_node_count = 1
    total_max_node_count = var.max_tpu_nodes
    location_policy      = "ANY"
  }
  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]
    machine_type = "ct6e-standard-4t"
  }
  management {
    auto_upgrade = true
    auto_repair  = true
  }
}

resource "helm_release" "arc" {
  name             = "actions-runner-controller"
  chart            = "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller"
  version          = "0.9.3"
  namespace        = var.arc_namespace
  create_namespace = true
}

resource "helm_release" "arc_runner_set" {
  name = "v6e-runner-set"
  depends_on = [
    helm_release.arc
  ]
  chart            = "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set"
  version          = "0.9.3"
  namespace        = var.runner_namespace
  create_namespace = true

  values = [
    templatefile("../terraform_modules/arc_v6e_container_cluster/arc-values.yaml", {
      github_repo_url = var.github_repo_url
      max_tpu_nodes   = var.max_tpu_nodes
      runner_image    = var.runner_image
    })
  ]
}
