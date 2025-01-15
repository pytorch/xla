# This Terraform configuration manages CI/CD infrastructure for PyTorch/XLA testing
# across multiple TPU hardware generations (v4, v5p, v6e). It creates:
# - Separate GKE clusters for each TPU version
# - Node pools with both CPU and TPU nodes
# - GitHub Actions runner configuration for automated testing
# - Custom CI runner container deployment
# 
# The infrastructure is used to run automated tests for the pytorch/xla repository
# ensuring compatibility and performance across TPU generations.

module "v4_arc_cluster" {
  source            = "../terraform_modules/arc_v4_container_cluster"
  project_id        = "tpu-pytorch-releases"
  cluster_name      = "tpu-ci"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 1
  tpu_nodepool_name = "tpu-nodepool"
  max_tpu_nodes     = 4
  github_repo_url   = "https://github.com/pytorch/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch-releases/tpu-ci-runner:latest"
}

module "v5p_arc_cluster" {
  source            = "../terraform_modules/arc_v5p_container_cluster"
  project_id        = "tpu-pytorch-releases"
  cluster_name      = "tpu-ci"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 1
  tpu_nodepool_name = "tpu-nodepool"
  max_tpu_nodes     = 4
  github_repo_url   = "https://github.com/pytorch/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch-releases/tpu-ci-runner:latest"
}

module "v6e_arc_cluster" {
  source            = "../terraform_modules/arc_v6e_container_cluster"
  project_id        = "tpu-pytorch-releases"
  cluster_name      = "tpu-ci"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 1
  tpu_nodepool_name = "tpu-nodepool"
  max_tpu_nodes     = 4
  github_repo_url   = "https://github.com/pytorch/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch-releases/tpu-ci-runner:latest"
}