module "v4_arc_cluster" {
  source            = "../terraform_modules/arc_v4_container_cluster"
  project_id        = "tpu-pytorch"
  cluster_name      = "tpu-ci"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 32
  tpu_nodepool_name = "tpu-nodepool"
  min_tpu_nodes     = 32
  max_tpu_nodes     = 32
  github_repo_url   = "https://github.com/pytorch/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch/tpu-ci-runner:latest"
}
