module "v4_arc_cluster" {
  source            = "../terraform_modules/arc_v4_container_cluster"
  project_id        = "tpu-pytorch-releases"
  cluster_name      = "tpu-ci"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 1
  tpu_nodepool_name = "tpu-nodepool"
  max_tpu_nodes     = 2
  github_repo_url   = "https://github.com/pytorch/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch-releases/tpu-ci-runner:latest"
}
