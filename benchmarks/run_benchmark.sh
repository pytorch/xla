#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/benchmark_test.log

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

TESTGPUVM=None
TESTTPUVM=None
# NUMBER=0

while getopts 'G:T:' OPTION # N:
do
  case $OPTION in
    G)
      TESTGPUVM=$OPTARG
      ;;
    T)
      TESTTPUVM=$OPTARG
      ;;
    # N)
    #   NUMBER=$OPTARG
    #   ;;
  esac
done
shift $(($OPTIND - 1))

# func for test after ssh to VM, create container and execute in container
function benchmarking_in_container {
  sudo docker pull gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8
  sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent    software-properties-common
  nvidia-smi
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  sudo docker run --gpus all -it -d gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8 bin/bash
  sudo docker exec -it $(sudo docker ps | awk 'NR==2 { print $1 }') /bin/bash
  # install torchbench
  cd ~
  git clone -b xla_benchmark https://github.com/pytorch/benchmark.git
  cd benchmark
  # install deps
  pip install --pre torchvision torchaudio -i https://download.pytorch.org/whl/nightly/cu118
  # git clone xla
  cd ~
  git clone -b benchmark https://github.com/pytorch/xla.git xla
  cd ~/xla/benchmarks
  # dry run
  python3 experiment_runner.py --suite-name=torchbench --accelerator=gpu --progress-bar --dry-run
  # run bechmark
  python3 experiment_runner.py --suite-name=torchbench --accelerator=gpu --progress-bar
  # analyze result to csv
  python3 result_analyzer.py
}



if TESTGPUVM='1A100':
  # ssh to 1-A100 GPUVM and test in container
  gcloud compute ssh a100-manfei-1 --zone us-central1-c --project tpu-prod-env-one-vm -- -o ProxyCommand='corp-ssh-helper %h %p' --command=benchmarking_in_container
elif TESTGPUVM='8A100':
  # SSH TO 8-A100 GPUVM and test in container
  gcloud compute ssh manfei-a100-8-new --zone us-central1-c --project tpu-prod-env-one-vm -- -o ProxyCommand='corp-ssh-helper %h %p' --command=benchmarking_in_container
elif TESTGPUVM='4H100':
  # ssh to 4-H100 GPUVM and test in container
elif TESTTPUVM='v5e8':
  # ssh to v5e-8 TPUVM and test in container
elif TESTTPUVM='v5p8':
  # ssh to v5p-8 TPUVM and test in container
