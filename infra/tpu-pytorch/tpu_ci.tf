module "v4_arc_cluster" {
  source            = "../terraform_modules/arc_v4_container_cluster"
  project_id        = "tpu-pytorch"
  cluster_name      = "tpu-ci-test"
  cpu_nodepool_name = "cpu-nodepool"
  cpu_node_count    = 1
  tpu_nodepool_name = "tpu-nodepool"
  max_tpu_nodes     = 1
  github_repo_url   = "https://github.com/mbzomowski-test-org/xla"
  # Dockerfile for this image can be found at test/tpu/Dockerfile
  runner_image      = "gcr.io/tpu-pytorch/tpu-ci-runner-test:latest"
}
